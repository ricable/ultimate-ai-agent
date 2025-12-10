"""
Interactive chat interface for Flow4 processed documents and models.

This module provides chat interfaces for interacting with processed documents
and fine-tuned models.
"""

import json
import os
import glob
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DocumentChatInterface:
    """Interactive chat interface for processed documents."""
    
    def __init__(self, chunks_dir: str = "output/chunks"):
        """Initialize chat interface with document chunks.
        
        Args:
            chunks_dir: Directory containing chunk files
        """
        self.chunks_dir = chunks_dir
        self.chunks = self.load_chunks()
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks from the Flow4 output directory."""
        chunks_path = Path(self.chunks_dir)
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        chunks = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load {chunk_file}: {e}")
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from {self.chunks_dir}")
        return chunks
    
    def simple_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based search through chunks."""
        query_words = query.lower().split()
        scored_chunks = []
        
        for chunk in self.chunks:
            text = chunk.get("text", "").lower()
            enriched_text = chunk.get("metadata", {}).get("enriched_text", "").lower()
            
            # Simple scoring based on keyword matches
            score = 0
            for word in query_words:
                score += text.count(word) * 2  # Weight original text higher
                score += enriched_text.count(word)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:max_results]]
    
    def format_chunk_for_display(self, chunk: Dict[str, Any], max_length: int = 500) -> str:
        """Format a chunk for display in the chat interface."""
        text = chunk.get("text", "")
        chunk_id = chunk.get("id", "unknown")
        source = chunk.get("source", "unknown")
        
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Get headings if available
        headings = chunk.get("metadata", {}).get("docling_headings", [])
        heading_str = f" - {headings[0]}" if headings else ""
        
        return f"[Chunk {chunk_id}{heading_str}]\n{text}\n"
    
    def generate_simple_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate a simple answer based on relevant chunks."""
        if not relevant_chunks:
            return "❌ No relevant information found in the documentation."
        
        answer = f"📚 Based on the documentation, here's what I found about '{query}':\n\n"
        
        for i, chunk in enumerate(relevant_chunks[:3], 1):  # Show top 3 chunks
            chunk_text = chunk.get("text", "")
            
            # Extract the most relevant sentence containing query words
            sentences = chunk_text.split('. ')
            query_words = query.lower().split()
            
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(sentence_lower.count(word) for word in query_words)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            if best_sentence:
                answer += f"{i}. {best_sentence.strip()}\n\n"
            else:
                # Fallback to first part of chunk
                first_part = chunk_text[:200].strip()
                answer += f"{i}. {first_part}...\n\n"
        
        answer += f"💡 Found {len(relevant_chunks)} relevant sections in the documentation."
        return answer
    
    def run_chat_interface(self):
        """Run the interactive chat interface."""
        print("\n" + "="*60)
        print("🤖 Flow4 Document Chat Interface")
        print("="*60)
        print("Ask questions about the technical documentation!")
        print("Type 'quit', 'exit', or 'q' to exit.")
        print("Type 'stats' to see document statistics.")
        print("Type 'help' for more commands.")
        print("-"*60)
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                    
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if not query:
                    print("Please enter a question!")
                    continue
                
                # Search for relevant chunks
                print("🔍 Searching documentation...")
                relevant_chunks = self.simple_search(query)
                
                if not relevant_chunks:
                    print("❌ No relevant information found. Try different keywords.")
                    continue
                
                # Generate and display answer
                answer = self.generate_simple_answer(query, relevant_chunks)
                print(f"\n{answer}")
                
                # Optionally show source chunks
                show_sources = input("\n📖 Show source chunks? (y/n): ").strip().lower()
                if show_sources in ['y', 'yes']:
                    print("\n" + "-"*40 + " SOURCE CHUNKS " + "-"*40)
                    for i, chunk in enumerate(relevant_chunks[:3], 1):
                        print(f"\n--- Chunk {i} ---")
                        print(self.format_chunk_for_display(chunk))
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def show_stats(self):
        """Show statistics about the loaded chunks."""
        total_chunks = len(self.chunks)
        total_chars = sum(len(chunk.get("text", "")) for chunk in self.chunks)
        
        # Count chunks by heading
        headings = {}
        for chunk in self.chunks:
            chunk_headings = chunk.get("metadata", {}).get("docling_headings", [])
            if chunk_headings:
                heading = chunk_headings[0]
                headings[heading] = headings.get(heading, 0) + 1
        
        print(f"\n📊 Document Statistics:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chunk size: {total_chars // total_chunks if total_chunks > 0 else 0} characters")
        print(f"   Unique headings: {len(headings)}")
        
        if headings:
            print(f"\n📋 Top sections:")
            for heading, count in sorted(headings.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {heading}: {count} chunks")
    
    def show_help(self):
        """Show help information."""
        print(f"\n💡 Available commands:")
        print(f"   • Ask any question about the documentation")
        print(f"   • 'stats' - Show document statistics")
        print(f"   • 'help' - Show this help")
        print(f"   • 'quit', 'exit', 'q' - Exit the chat")
        print(f"\n🔍 Search tips:")
        print(f"   • Use specific technical terms")
        print(f"   • Try acronyms (e.g., 'BWP', 'DCI', 'RRC')")
        print(f"   • Ask about specific features or procedures")
    
    def export_qa_dataset(self, output_file: str):
        """Export a simple Q&A dataset for fine-tuning."""
        logger.info(f"📤 Generating Q&A dataset from {len(self.chunks)} chunks...")
        
        qa_pairs = []
        
        for chunk in self.chunks:
            text = chunk.get("text", "").strip()
            if len(text) < 50:  # Skip very short chunks
                continue
            
            headings = chunk.get("metadata", {}).get("docling_headings", [])
            
            # Generate simple Q&A pairs based on the content
            if headings:
                # Question about the section
                qa_pairs.append({
                    "question": f"What is {headings[0]} about?",
                    "answer": text[:500] + ("..." if len(text) > 500 else ""),
                    "metadata": {
                        "chunk_id": chunk.get("id"),
                        "heading": headings[0],
                        "type": "section_overview"
                    }
                })
            
            # Look for specific features or technical terms
            if any(term in text.lower() for term in ["feature", "function", "parameter", "setting"]):
                # Extract first sentence as a potential answer
                sentences = text.split('. ')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 20:
                        qa_pairs.append({
                            "question": f"Can you explain this technical concept?",
                            "answer": first_sentence,
                            "metadata": {
                                "chunk_id": chunk.get("id"),
                                "type": "technical_explanation"
                            }
                        })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported {len(qa_pairs)} Q&A pairs to {output_file}")
        logger.info(f"💡 This dataset can be used for fine-tuning language models.")


class ModelChatInterface:
    """Interactive chat interface for fine-tuned models."""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 adapter_path: str = "fine_tuned_adapters"):
        """Initialize model chat interface.
        
        Args:
            model_name: Base model name
            adapter_path: Path to fine-tuned adapters
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
    
    def generate_response(self, prompt: str, max_tokens: int = 150, 
                         temperature: float = 0.7) -> str:
        """Generate response using MLX model."""
        # Format as Q&A prompt (matching training format)
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "mlx_lm.generate",
                "--model", self.model_name,
                "--adapter-path", self.adapter_path,
                "--prompt", formatted_prompt,
                "--max-tokens", str(max_tokens),
                "--temp", str(temperature)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract the generated text from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('=========='):
                        continue
                    if 'tokens-per-sec' in line or 'Peak memory' in line or 'Prompt:' in line:
                        continue
                    if line.strip() and not line.startswith('Calling') and not line.startswith('Fetching'):
                        return line.strip()
                return "Response generated successfully but could not extract text."
            else:
                return f"Error generating response: {result.stderr}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def run_chat_interface(self):
        """Run the interactive model chat interface."""
        print("🎯 Flow4 Fine-tuned Model Chat Interface")
        print("=" * 50)
        print(f"Model: {self.model_name} + LoRA adapters")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                print("🤖 AI: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")