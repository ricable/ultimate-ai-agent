#!/usr/bin/env python3
"""
HuggingFace CLI Chat Interface
=============================

Interactive command-line chat interface for HuggingFace models with:
- MPS acceleration support
- Streaming responses
- Chat history management
- Multiple model support
- Quantization options
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime

try:
    import flow2
    if not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace framework not available")
        print("Install with: pip install flow2[huggingface]")
        sys.exit(1)
except ImportError:
    print("❌ Flow2 not available")
    print("Install with: pip install flow2")
    sys.exit(1)

class HuggingFaceChatInterface:
    """Interactive chat interface for HuggingFace models."""
    
    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "auto",
        system_message: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        streaming: bool = True
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.device_name = device
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.streaming = streaming
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.chat_history: List[Dict[str, str]] = []
        self.session_start = datetime.now()
        
        # Load model
        self._load_model()
        
        # Setup generation parameters
        self.gen_params = flow2.frameworks.huggingface.GenerationParams(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def _load_model(self):
        """Load the HuggingFace model."""
        print(f"🤗 Loading HuggingFace model: {self.model_name}")
        
        # Setup device
        if self.device_name == "auto":
            self.device = flow2.frameworks.huggingface.setup_mps_device()
        else:
            import torch
            self.device = torch.device(self.device_name)
        
        print(f"🔧 Device: {self.device}")
        
        # Setup quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
            print("🔧 Using 4-bit quantization")
        elif self.quantization == "8bit":
            quantization_config = {
                "load_in_8bit": True
            }
            print("🔧 Using 8-bit quantization")
        
        # Load model
        start_time = time.time()
        try:
            self.model, self.tokenizer = flow2.frameworks.huggingface.load_hf_model(
                self.model_name,
                device=self.device,
                quantization_config=quantization_config
            )
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            # Print model info
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 Parameters: {flow2.frameworks.huggingface.format_model_size(num_params)}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
    
    def _format_chat_history(self) -> List[Dict[str, str]]:
        """Format chat history for the model."""
        messages = []
        
        # Add system message if provided
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        # Add chat history
        messages.extend(self.chat_history)
        
        return messages
    
    def _save_chat_history(self, filename: Optional[str] = None):
        """Save chat history to file."""
        if not filename:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            filename = f"hf_chat_history_{timestamp}.json"
        
        chat_data = {
            "model": self.model_name,
            "quantization": self.quantization,
            "device": str(self.device),
            "system_message": self.system_message,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "chat_history": self.chat_history,
            "generation_params": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(chat_data, f, indent=2)
            print(f"💾 Chat history saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save chat history: {e}")
    
    def _load_chat_history(self, filename: str):
        """Load chat history from file."""
        try:
            with open(filename, 'r') as f:
                chat_data = json.load(f)
            
            self.chat_history = chat_data.get("chat_history", [])
            self.system_message = chat_data.get("system_message")
            
            print(f"📂 Loaded {len(self.chat_history)} messages from {filename}")
            
            # Print recent messages
            if self.chat_history:
                print("\n📜 Recent conversation:")
                for msg in self.chat_history[-4:]:  # Show last 4 messages
                    role = msg["role"].title()
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    print(f"  {role}: {content}")
                print()
            
        except Exception as e:
            print(f"❌ Failed to load chat history: {e}")
    
    def _print_help(self):
        """Print help information."""
        print("\n🤗 HuggingFace Chat Commands:")
        print("  /help         - Show this help message")
        print("  /history      - Show recent chat history")
        print("  /save [file]  - Save chat history to file")
        print("  /load <file>  - Load chat history from file")
        print("  /clear        - Clear chat history")
        print("  /system <msg> - Set system message")
        print("  /temp <value> - Set temperature (0.1-2.0)")
        print("  /tokens <num> - Set max tokens (1-2048)")
        print("  /stream       - Toggle streaming mode")
        print("  /info         - Show model and session info")
        print("  /quit, /exit  - Exit the chat")
        print()
    
    def _print_info(self):
        """Print model and session information."""
        print(f"\n🤗 Session Information:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Quantization: {self.quantization or 'None'}")
        print(f"  System Message: {self.system_message or 'None'}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  Streaming: {self.streaming}")
        print(f"  Messages in History: {len(self.chat_history)}")
        print(f"  Session Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def chat_loop(self):
        """Main chat interaction loop."""
        print("\n🤗 HuggingFace Chat Interface")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        if self.quantization:
            print(f"Quantization: {self.quantization}")
        print("\nType your message or '/help' for commands.")
        print("Use '/quit' or '/exit' to end the session.\n")
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("👤 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit']:
                        print("👋 Goodbye!")
                        break
                    elif user_input == '/help':
                        self._print_help()
                        continue
                    elif user_input == '/history':
                        self._print_recent_history()
                        continue
                    elif user_input.startswith('/save'):
                        parts = user_input.split(' ', 1)
                        filename = parts[1] if len(parts) > 1 else None
                        self._save_chat_history(filename)
                        continue
                    elif user_input.startswith('/load'):
                        parts = user_input.split(' ', 1)
                        if len(parts) > 1:
                            self._load_chat_history(parts[1])
                        else:
                            print("❌ Please specify a filename: /load <filename>")
                        continue
                    elif user_input == '/clear':
                        self.chat_history.clear()
                        print("🧹 Chat history cleared.")
                        continue
                    elif user_input.startswith('/system'):
                        parts = user_input.split(' ', 1)
                        if len(parts) > 1:
                            self.system_message = parts[1]
                            print(f"⚙️  System message set: {self.system_message}")
                        else:
                            self.system_message = None
                            print("⚙️  System message cleared.")
                        continue
                    elif user_input.startswith('/temp'):
                        parts = user_input.split(' ', 1)
                        if len(parts) > 1:
                            try:
                                temp = float(parts[1])
                                if 0.1 <= temp <= 2.0:
                                    self.temperature = temp
                                    self.gen_params.temperature = temp
                                    print(f"🌡️  Temperature set to {temp}")
                                else:
                                    print("❌ Temperature must be between 0.1 and 2.0")
                            except ValueError:
                                print("❌ Invalid temperature value")
                        else:
                            print("❌ Please specify temperature: /temp <value>")
                        continue
                    elif user_input.startswith('/tokens'):
                        parts = user_input.split(' ', 1)
                        if len(parts) > 1:
                            try:
                                tokens = int(parts[1])
                                if 1 <= tokens <= 2048:
                                    self.max_tokens = tokens
                                    self.gen_params.max_new_tokens = tokens
                                    print(f"📝 Max tokens set to {tokens}")
                                else:
                                    print("❌ Max tokens must be between 1 and 2048")
                            except ValueError:
                                print("❌ Invalid token count")
                        else:
                            print("❌ Please specify token count: /tokens <number>")
                        continue
                    elif user_input == '/stream':
                        self.streaming = not self.streaming
                        print(f"🌊 Streaming {'enabled' if self.streaming else 'disabled'}")
                        continue
                    elif user_input == '/info':
                        self._print_info()
                        continue
                    else:
                        print("❌ Unknown command. Type '/help' for available commands.")
                        continue
                
                # Add user message to history
                self.chat_history.append({"role": "user", "content": user_input})
                
                # Generate assistant response
                print("🤖 Assistant: ", end="", flush=True)
                
                start_time = time.time()
                
                try:
                    if self.streaming:
                        # Streaming response
                        response = ""
                        messages = self._format_chat_history()
                        
                        for token in flow2.frameworks.huggingface.streaming_completion(
                            self.model, self.tokenizer, 
                            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(self.tokenizer, 'apply_chat_template') else user_input,
                            self.gen_params, print_output=False
                        ):
                            response += token
                            print(token, end="", flush=True)
                        
                        print()  # New line after streaming
                    else:
                        # Regular response
                        messages = self._format_chat_history()
                        response = flow2.frameworks.huggingface.chat_completion(
                            self.model, self.tokenizer, messages, self.gen_params, self.system_message
                        )
                        print(response)
                    
                    # Add assistant response to history
                    self.chat_history.append({"role": "assistant", "content": response})
                    
                    # Print timing info
                    generation_time = time.time() - start_time
                    print(f"⏱️  ({generation_time:.2f}s)")
                    
                except Exception as e:
                    print(f"❌ Error generating response: {e}")
                
                print()  # Empty line for readability
        
        except Exception as e:
            print(f"❌ Chat error: {e}")
        
        finally:
            # Offer to save chat history
            if self.chat_history:
                try:
                    save_response = input("\n💾 Save chat history? (y/N): ").strip().lower()
                    if save_response in ['y', 'yes']:
                        self._save_chat_history()
                except (EOFError, KeyboardInterrupt):
                    pass
    
    def _print_recent_history(self, count: int = 10):
        """Print recent chat history."""
        if not self.chat_history:
            print("📜 No chat history available.")
            return
        
        print(f"\n📜 Recent Chat History (last {min(count, len(self.chat_history))} messages):")
        print("-" * 50)
        
        recent_history = self.chat_history[-count:]
        for i, msg in enumerate(recent_history, 1):
            role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
            content = msg["content"]
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."
            
            print(f"{i:2d}. {role}: {content}")
        
        print()

def main():
    parser = argparse.ArgumentParser(description="HuggingFace CLI Chat Interface")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                       help="Model name or path")
    parser.add_argument("--quantization", choices=["4bit", "8bit"],
                       help="Use quantization")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto",
                       help="Device to use")
    parser.add_argument("--system-message", type=str,
                       help="System message for the conversation")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--no-streaming", action="store_true",
                       help="Disable streaming output")
    parser.add_argument("--load-history", type=str,
                       help="Load chat history from file")
    
    args = parser.parse_args()
    
    # Create chat interface
    chat = HuggingFaceChatInterface(
        model_name=args.model,
        quantization=args.quantization,
        device=args.device,
        system_message=args.system_message,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        streaming=not args.no_streaming
    )
    
    # Load history if specified
    if args.load_history:
        chat._load_chat_history(args.load_history)
    
    # Start chat loop
    chat.chat_loop()

if __name__ == "__main__":
    main()