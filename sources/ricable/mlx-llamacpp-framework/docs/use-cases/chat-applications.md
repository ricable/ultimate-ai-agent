# Building Chat Applications with LLMs on Apple Silicon

This guide covers how to build interactive chat applications using large language models on Apple Silicon hardware with both llama.cpp and MLX frameworks.

## Table of Contents
- [Understanding Chat Interfaces](#understanding-chat-interfaces)
- [llama.cpp Chat Solutions](#llamacpp-chat-solutions)
- [MLX Chat Solutions](#mlx-chat-solutions)
- [Advanced Chat Features](#advanced-chat-features)
- [Web Interfaces](#web-interfaces)
- [Performance Optimization](#performance-optimization)
- [Example Projects](#example-projects)

## Understanding Chat Interfaces

### Chat vs. Regular Inference

Chat interfaces differ from basic inference in several ways:
- They maintain a conversation history
- They use specific formatting for different speakers
- They often include system prompts to guide behavior
- They require proper handling of context length as conversations grow

### Chat Templates

Chat models use templates to format multi-turn conversations. Common formats include:

1. **Basic Alternating Format**:
   ```
   User: {user_message}
   Assistant: {assistant_message}
   User: {next_user_message}
   ```

2. **Chat Markup Language (ChatML)**:
   ```
   <|im_start|>system
   You are a helpful assistant.
   <|im_end|>
   <|im_start|>user
   Hello, who are you?
   <|im_end|>
   <|im_start|>assistant
   ```

3. **Alpaca/Vicuna Format**:
   ```
   ### Instruction:
   {user_message}

   ### Response:
   {assistant_message}
   ```

### Token Management

Effective token management is critical for chat applications:
- Track token usage throughout the conversation
- Remove older messages when approaching context limits
- Reserve tokens for both the response and future messages

## llama.cpp Chat Solutions

### Basic Interactive Chat

```bash
# Simple interactive mode
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive --color
```

### Custom Chat Template

```bash
# Create a chat template file
cat > chat_template.txt << EOL
<|im_start|>system
You are a helpful, concise, and accurate assistant.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
EOL

# Use the template
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive --color -f chat_template.txt
```

### Custom Chat Script (C++)

```cpp
// Simple chat application (chat.cpp)
#include <iostream>
#include <string>
#include <vector>
#include "llama.h"

int main(int argc, char** argv) {
    // Initialize model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1; // Use all layers on GPU
    
    llama_model* model = llama_load_model_from_file("models/llama-2-7b-q4_k.gguf", model_params);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    // Initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    
    // Chat loop
    std::string history = "You are a helpful assistant.\n\n";
    std::cout << "Assistant: Hello! How can I help you today?\n";
    
    std::string user_input;
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, user_input);
        
        if (user_input == "exit" || user_input == "quit") {
            break;
        }
        
        // Update history with user input
        history += "User: " + user_input + "\nAssistant: ";
        
        // Generate response
        std::cout << "\nAssistant: ";
        
        const int max_tokens = 512;
        const float temperature = 0.7f;
        const float top_p = 0.95f;
        const float repeat_penalty = 1.1f;
        
        llama_batch batch = llama_batch_init(history.size() + max_tokens, 0, 1);
        
        // Tokenize the prompt
        for (int i = 0; i < history.size(); ++i) {
            batch.token[i] = llama_token_bos();  // Simplified
        }
        batch.n_tokens = history.size();
        
        // Generate tokens
        std::string response;
        for (int i = 0; i < max_tokens; ++i) {
            llama_decode(ctx, batch);
            
            // Get next token
            llama_token token = llama_sample_token(ctx);
            
            if (token == llama_token_eos()) {
                break;
            }
            
            // Convert token to text
            const char* token_str = llama_token_to_str(ctx, token);
            std::cout << token_str;
            response += token_str;
            
            // Update batch for next iteration
            batch.token[0] = token;
            batch.n_tokens = 1;
        }
        
        // Update history with response
        history += response + "\n";
    }
    
    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    
    return 0;
}
```

### Python Chat Interface

```python
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="models/llama-2-7b-q4_k.gguf",
    n_ctx=2048,
    n_gpu_layers=-1  # Use all layers on GPU
)

# Chat history format
system_message = "You are a helpful, concise, and accurate assistant."
messages = [{"role": "system", "content": system_message}]

print("Assistant: Hello! How can I help you today?")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Add user message to history
    messages.append({"role": "user", "content": user_input})
    
    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1
    )
    
    # Extract and print response
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"\nAssistant: {assistant_message}")
    
    # Add assistant response to history
    messages.append({"role": "assistant", "content": assistant_message})
    
    # Check token count and truncate history if needed
    if len(llm.tokenize("".join([m["content"] for m in messages]))) > 1536:
        # Keep system message and last 4 exchanges
        messages = [messages[0]] + messages[-8:]
```

### Server with Chat API

```bash
# Run server with chat capability
./server -m models/llama-2-7b-q4_k.gguf --metal -c 2048 --host 0.0.0.0 --port 8080
```

```python
# Python client to connect to server
import requests
import json

url = "http://localhost:8080/chat"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Apple Silicon?"}
]

response = requests.post(
    url,
    json={
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512
    }
)

print(json.loads(response.text)["choices"][0]["message"]["content"])
```

## MLX Chat Solutions

### Basic Chat Implementation

```python
from mlx_lm import load
import mlx.core as mx

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Chat function
def chat_with_model():
    # System prompt
    history = "You are a helpful assistant.\n\n"
    print("Assistant: Hello! How can I help you today?")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Update history
        prompt = history + f"User: {user_input}\nAssistant: "
        history = prompt
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        
        tokens = []
        for token in generate(model, tokenizer, prompt, max_tokens=512, temp=0.7, stream=True):
            print(token, end="", flush=True)
            tokens.append(token)
        
        response = "".join(tokens)
        history += response + "\n"

# Run the chat interface
chat_with_model()
```

### Advanced Chat Implementation

```python
import mlx.core as mx
from mlx_lm import load, generate
import json
import os

class ChatInterface:
    def __init__(self, model_path, quantization="int4"):
        self.model, self.tokenizer = load(model_path, quantization=quantization)
        self.system_prompt = "You are a helpful, concise, and accurate assistant."
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.max_context_length = 2048
        
    def format_prompt(self):
        formatted = ""
        for msg in self.messages:
            if msg["role"] == "system":
                formatted += f"<|im_start|>system\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "user":
                formatted += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "assistant":
                formatted += f"<|im_start|>assistant\n{msg['content']}\n<|im_end|>\n"
        
        # Add final assistant prefix for the response
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def get_response(self, user_input, temperature=0.7):
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Format the prompt
        prompt = self.format_prompt()
        
        # Check token length and truncate if needed
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.max_context_length - 512:  # Reserve space for response
            # Keep system message and last few exchanges
            self.messages = [self.messages[0]] + self.messages[-6:]
            prompt = self.format_prompt()
        
        # Generate response
        response_tokens = []
        for token in generate(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_tokens=512, 
            temp=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
            stream=True
        ):
            response_tokens.append(token)
            yield token
        
        # Add assistant message to history
        full_response = "".join(response_tokens)
        self.messages.append({"role": "assistant", "content": full_response})
    
    def save_conversation(self, filename):
        with open(filename, "w") as f:
            json.dump(self.messages, f, indent=2)
    
    def load_conversation(self, filename):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.messages = json.load(f)

# Example usage
chat = ChatInterface("llama-2-7b")

print("Assistant: Hello! How can I help you today?")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    print("\nAssistant: ", end="", flush=True)
    for token in chat.get_response(user_input):
        print(token, end="", flush=True)
    print()
    
    # Optional: Save conversation
    # chat.save_conversation("conversation.json")
```

### MLX Chat Server

```python
from flask import Flask, request, jsonify
from mlx_lm import load, generate
import mlx.core as mx
import threading

app = Flask(__name__)

# Load model (do this at startup)
print("Loading model...")
model, tokenizer = load("llama-2-7b", quantization="int4")
print("Model loaded")

# Simple in-memory conversation store
conversations = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    conversation_id = data.get('conversation_id', 'default')
    user_message = data.get('message', '')
    temperature = data.get('temperature', 0.7)
    
    # Get or create conversation
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    
    # Add user message
    conversations[conversation_id].append({"role": "user", "content": user_message})
    
    # Format prompt
    prompt = ""
    for msg in conversations[conversation_id]:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    
    prompt += "Assistant: "
    
    # Generate response
    response_tokens = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=512,
        temp=temperature
    )
    response_text = tokenizer.decode(response_tokens)
    
    # Add to conversation history
    conversations[conversation_id].append({"role": "assistant", "content": response_text})
    
    # Trim conversation if too long
    if len(conversations[conversation_id]) > 12:  # Keep system + 5 exchanges
        conversations[conversation_id] = [conversations[conversation_id][0]] + conversations[conversation_id][-10:]
    
    return jsonify({
        "conversation_id": conversation_id,
        "response": response_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Advanced Chat Features

### Memory Management

```python
class ConversationMemory:
    def __init__(self, tokenizer, max_tokens=2048):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.messages = []
        self.system_prompt = "You are a helpful assistant."
        self.token_counts = {}
    
    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.messages.append(message)
        
        # Count tokens
        token_count = len(self.tokenizer.encode(content))
        self.token_counts[len(self.messages) - 1] = token_count
        
        # Prune if necessary
        self._prune_messages()
    
    def _prune_messages(self):
        # Always keep system prompt and latest user message
        total_tokens = sum(self.token_counts.values())
        
        if total_tokens > self.max_tokens - 512:  # Reserve space for response
            # Start removing old messages, but keep system prompt
            i = 1  # Start after system prompt
            while total_tokens > self.max_tokens - 512 and i < len(self.messages) - 1:
                total_tokens -= self.token_counts[i]
                self.messages[i] = {"role": self.messages[i]["role"], "content": "[Truncated message]"}
                self.token_counts[i] = 14  # Approx tokens for truncated placeholder
                i += 1
    
    def get_formatted_prompt(self):
        # Format for model input
        formatted = f"System: {self.system_prompt}\n\n"
        
        for msg in self.messages:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"Assistant: {msg['content']}\n"
        
        formatted += "Assistant: "
        return formatted
```

### Conversation Summarization

When conversations get too long, use the model to summarize previous turns:

```python
def summarize_conversation(model, tokenizer, messages, max_summary_tokens=256):
    # Format conversation for summarization
    conversation = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        conversation += f"{role.capitalize()}: {content}\n"
    
    # Create summarization prompt
    prompt = f"Below is a conversation between a user and an assistant. Summarize the key points of this conversation in a concise way:\n\n{conversation}\n\nSummary:"
    
    # Generate summary
    summary_tokens = generate(model, tokenizer, prompt, max_tokens=max_summary_tokens)
    summary = tokenizer.decode(summary_tokens)
    
    return summary

# Usage in conversation manager
if len(conversation.messages) > 10:
    summary = summarize_conversation(model, tokenizer, conversation.messages[1:-2])
    conversation.messages = [
        conversation.messages[0],  # Keep system prompt
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        conversation.messages[-2],  # Keep last user message
        conversation.messages[-1]   # Keep last assistant message
    ]
```

### Chat Personalities

Use different system prompts to create different chat personalities:

```python
PERSONALITIES = {
    "helpful": "You are a helpful, accurate, and friendly assistant. You provide detailed and thoughtful responses.",
    "concise": "You are a concise and direct assistant. Provide brief, to-the-point answers without unnecessary details.",
    "expert": "You are an expert technical assistant with deep knowledge of programming, AI, and computer science. Use technical terminology appropriately.",
    "creative": "You are a creative assistant with a vivid imagination. Your responses should be colorful, engaging, and think outside the box.",
    "teacher": "You are a patient teacher who explains concepts clearly. Break down complex topics into simple terms and use analogies to aid understanding."
}

# Usage
personality = "expert"  # User selected
system_prompt = PERSONALITIES[personality]
```

## Web Interfaces

### Simple HTML/JS Chat Interface

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #chat-container { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .user-message { background-color: #e1f5fe; padding: 8px; border-radius: 10px; margin: 5px 0; }
        .assistant-message { background-color: #f5f5f5; padding: 8px; border-radius: 10px; margin: 5px 0; }
        #user-input { width: 80%; padding: 8px; }
        button { padding: 8px 15px; }
    </style>
</head>
<body>
    <h1>Chat with LLM</h1>
    <div id="chat-container"></div>
    <div>
        <input type="text" id="user-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        let conversationId = Date.now().toString();
        
        // Add initial assistant message
        addMessage('Hello! How can I help you today?', 'assistant');
        
        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = sender + '-message';
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            userInput.value = '';
            
            // Add loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'assistant-message';
            loadingDiv.textContent = 'Thinking...';
            chatContainer.appendChild(loadingDiv);
            
            try {
                // Send to API
                const response = await fetch('http://localhost:8080/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        conversation_id: conversationId,
                        message: message,
                        temperature: 0.7
                    })
                });
                
                const data = await response.json();
                
                // Remove loading indicator
                chatContainer.removeChild(loadingDiv);
                
                // Add assistant response
                addMessage(data.response, 'assistant');
            } catch (error) {
                chatContainer.removeChild(loadingDiv);
                addMessage('Error: Could not connect to the server.', 'assistant');
                console.error(error);
            }
        }
        
        // Allow sending message with Enter key
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

### Using Gradio for UI

```python
import gradio as gr
from mlx_lm import load, generate
import mlx.core as mx

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Chat state
class ChatState:
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.max_tokens = 2048

chat_state = ChatState()

def format_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    prompt += "Assistant: "
    return prompt

def chat(message, history):
    # Add user message
    chat_state.messages.append({"role": "user", "content": message})
    
    # Format prompt
    prompt = format_prompt(chat_state.messages)
    
    # Check length and truncate if needed
    if len(tokenizer.encode(prompt)) > chat_state.max_tokens - 512:
        chat_state.messages = [chat_state.messages[0]] + chat_state.messages[-6:]
        prompt = format_prompt(chat_state.messages)
    
    # Generate response
    response_tokens = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=512,
        temp=0.7
    )
    response_text = tokenizer.decode(response_tokens)
    
    # Add to history
    chat_state.messages.append({"role": "assistant", "content": response_text})
    
    return response_text

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="Chat with LLM on Apple Silicon",
    description="Powered by llama-2-7b with MLX"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Performance Optimization

### Memory Efficiency

1. **Minimize History Storage**:
   - Store only essential parts of the conversation
   - Truncate or summarize older parts when needed

2. **Token-based Memory Management**:
   - Track token counts for each message
   - Prioritize removing older messages

3. **Streaming Responses**:
   - Generate tokens one by one instead of all at once
   - Reduces peak memory usage

### Speed Optimization

1. **Batch Processing**:
   - Use larger batch sizes for decoding (e.g., 8-32)
   - Process multiple response tokens at once

2. **Reduced Precision Inference**:
   - Use INT4 quantization for faster inference
   - Balance between quality and speed

3. **Precompute and Cache**:
   - Cache system prompt encoding
   - Cache frequently used responses

## Example Projects

### Standalone Desktop Chat App

```python
import tkinter as tk
from tkinter import scrolledtext
import threading
from llama_cpp import Llama

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Chat App")
        self.root.geometry("600x800")
        
        # Create UI
        self.create_widgets()
        
        # Load model in background
        self.loading_label.config(text="Loading model... Please wait.")
        threading.Thread(target=self.load_model).start()
    
    def create_widgets(self):
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # User input area
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
        
        self.user_input = tk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)
        
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Loading indicator
        self.loading_label = tk.Label(self.root, text="")
        self.loading_label.pack(pady=(0, 10))
    
    def load_model(self):
        try:
            self.llm = Llama(
                model_path="models/llama-2-7b-q4_k.gguf",
                n_ctx=2048,
                n_gpu_layers=-1
            )
            self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Update UI
            self.root.after(0, lambda: self.loading_label.config(text="Model loaded. Ready to chat!"))
            self.root.after(3000, lambda: self.loading_label.config(text=""))
            
            # Add initial message
            self.add_message("Hello! How can I help you today?", "Assistant")
        except Exception as e:
            self.root.after(0, lambda: self.loading_label.config(text=f"Error: {str(e)}"))
    
    def add_message(self, message, sender):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
    
    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if not message:
            return
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
        # Add user message
        self.add_message(message, "You")
        
        # Disable input while processing
        self.user_input.config(state='disabled')
        self.send_button.config(state='disabled')
        self.loading_label.config(text="Thinking...")
        
        # Process in background
        threading.Thread(target=self.process_message, args=(message,)).start()
    
    def process_message(self, message):
        try:
            # Add to history
            self.messages.append({"role": "user", "content": message})
            
            # Generate response
            response = self.llm.create_chat_completion(
                messages=self.messages,
                max_tokens=512,
                temperature=0.7
            )
            
            assistant_message = response["choices"][0]["message"]["content"]
            
            # Add to history
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # Update UI
            self.root.after(0, lambda: self.add_message(assistant_message, "Assistant"))
            self.root.after(0, lambda: self.loading_label.config(text=""))
            
            # Truncate history if needed
            if len(self.messages) > 12:
                self.messages = [self.messages[0]] + self.messages[-10:]
        except Exception as e:
            self.root.after(0, lambda: self.add_message(f"Error: {str(e)}", "System"))
        finally:
            # Re-enable input
            self.root.after(0, lambda: self.user_input.config(state='normal'))
            self.root.after(0, lambda: self.send_button.config(state='normal'))
            self.root.after(0, lambda: self.user_input.focus())

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
```

### Chat Bot Integration

Example of integrating with other applications via a simple API:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from mlx_lm import load, generate

app = FastAPI()

# Load model at startup
model, tokenizer = load("llama-2-7b", quantization="int4")

# Data models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str
    tokens_used: int

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Format prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    
    prompt += "Assistant: "
    
    # Generate response
    try:
        response_tokens = generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=request.max_tokens,
            temp=request.temperature
        )
        response_text = tokenizer.decode(response_tokens)
        
        # Calculate tokens used
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(response_tokens)
        total_tokens = input_tokens + output_tokens
        
        return ChatResponse(
            response=response_text,
            tokens_used=total_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Further Reading

- [Improving Chat Quality](../chat_interfaces/docs/improving_chat_quality.md) - Advanced techniques for better responses
- [Fine-tuning for Chat](fine-tuning-guide.md) - Customize models for better chat performance
- [Hardware Recommendations](../hardware/hardware-recommendations.md) - Choose the right Mac for chat applications