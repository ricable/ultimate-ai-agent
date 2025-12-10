#!/usr/bin/env python3
"""
Web UI for MLX Chat Interface

This script provides a simple web interface for interacting with MLX models.
It uses Flask for the web server and includes WebSocket support for streaming responses.
"""

import os
import sys
import json
import argparse
import threading
import time
import queue
from typing import List, Dict, Optional, Any, Union, Tuple

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from common.chat_history import create_chat_session, ChatHistory

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention optimizations available")
except ImportError:
    print("⚠️  Flash Attention not available, using standard MLX attention")
    FLASH_ATTENTION_AVAILABLE = False

# Flask and related imports
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO

# Setup Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mlx-web-ui-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
MODELS = []
DEFAULT_MODEL = ""
ACTIVE_SESSIONS = {}  # Dictionary to store active chat sessions
LOADED_MODELS = {}    # Cache for loaded models to avoid reloading
FLASH_ATTENTION_ENABLED = True  # Global Flash Attention setting


def check_mlx_installation() -> bool:
    """
    Check if MLX is installed.
    
    Returns:
        Boolean indicating if MLX is installed
    """
    try:
        import mlx
        import mlx.core
        return True
    except ImportError:
        return False


def list_available_models() -> List[str]:
    """
    List available MLX model directories.
    
    Returns:
        List of model paths
    """
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../mlx-setup/models'
    ))
    
    if not os.path.exists(models_dir):
        return []
    
    # Return directories that contain a config.json file
    return [
        os.path.join(models_dir, d) 
        for d in os.listdir(models_dir) 
        if os.path.isdir(os.path.join(models_dir, d)) and 
           os.path.exists(os.path.join(models_dir, d, 'config.json'))
    ]


def apply_flash_attention_to_model(model, use_flash_attention=True, block_size=None):
    """
    Apply Flash Attention optimizations to model attention layers
    """
    if not use_flash_attention or not FLASH_ATTENTION_AVAILABLE:
        print("ℹ️ Using standard MLX attention")
        return model, 0
    
    print("🚀 Applying Flash Attention optimizations...")
    attention_replacements = 0
    
    def replace_attention_recursive(module, name_prefix=""):
        nonlocal attention_replacements
        
        # Handle MLX models which may have different attribute access patterns
        try:
            for name in dir(module):
                if name.startswith('_') or name in ['training', 'parameters', 'modules']:
                    continue
                    
                try:
                    child = getattr(module, name)
                    if not hasattr(child, '__class__'):
                        continue
                        
                    full_name = f"{name_prefix}.{name}" if name_prefix else name
                    
                    # Check if this is an attention layer we should replace
                    if hasattr(child, '__class__') and 'MultiHeadAttention' in str(child.__class__):
                        print(f"🔄 Replacing {full_name} with Flash Attention")
                        
                        # Create optimized replacement
                        try:
                            flash_attention = OptimizedMLXMultiHeadAttention(
                                child.dims,
                                child.num_heads,
                                bias=hasattr(child, 'bias'),
                                use_flash_attention=True,
                                block_size=block_size
                            )
                            
                            # Copy weights from original layer
                            if hasattr(child, 'q_proj') and hasattr(child.q_proj, 'weight'):
                                flash_attention.q_proj.weight = child.q_proj.weight
                                flash_attention.k_proj.weight = child.k_proj.weight  
                                flash_attention.v_proj.weight = child.v_proj.weight
                                flash_attention.out_proj.weight = child.out_proj.weight
                                
                                if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                                    flash_attention.q_proj.bias = child.q_proj.bias
                                    flash_attention.k_proj.bias = child.k_proj.bias
                                    flash_attention.v_proj.bias = child.v_proj.bias
                                    flash_attention.out_proj.bias = child.out_proj.bias
                            
                            # Replace the layer
                            setattr(module, name, flash_attention)
                            attention_replacements += 1
                        except Exception as e:
                            print(f"⚠️ Failed to replace {full_name}: {e}")
                    else:
                        # Recursively process child modules
                        replace_attention_recursive(child, full_name)
                        
                except (AttributeError, TypeError):
                    continue
                    
        except (AttributeError, TypeError):
            pass
    
    try:
        replace_attention_recursive(model)
        
        if attention_replacements > 0:
            print(f"✅ Replaced {attention_replacements} attention layers with Flash Attention")
        else:
            print("ℹ️ No compatible attention layers found for replacement")
            
    except Exception as e:
        print(f"⚠️ Flash Attention integration failed: {e}")
        print("ℹ️ Continuing with standard MLX attention")
    
    return model, attention_replacements


def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load an MLX model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    global LOADED_MODELS
    
    # Check if model is already loaded
    if model_path in LOADED_MODELS:
        return LOADED_MODELS[model_path]
    
    try:
        # Import MLX modules
        from mlx_lm import load
        
        # Load the model
        print(f"Loading model from {model_path}...")
        model, tokenizer = load(model_path)
        print(f"Model loaded successfully!")
        
        # Apply Flash Attention optimizations
        if FLASH_ATTENTION_ENABLED and FLASH_ATTENTION_AVAILABLE:
            model, flash_replacements = apply_flash_attention_to_model(
                model, 
                use_flash_attention=FLASH_ATTENTION_ENABLED
            )
            if flash_replacements > 0:
                print(f"📊 Flash Attention: {flash_replacements} layers optimized")
        
        # Cache the model
        LOADED_MODELS[model_path] = (model, tokenizer)
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def generate_mlx_response_stream(
    model_path: str, 
    messages: List[Dict[str, str]], 
    session_id: str,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> None:
    """
    Run the MLX model and stream output via WebSocket.
    
    Args:
        model_path: Path to the model directory
        messages: List of chat messages
        session_id: Session ID for WebSocket communication
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
    """
    try:
        # Import MLX modules
        import mlx.core as mx
        from mlx_lm import generate
        
        # Load the model
        model, tokenizer = load_model(model_path)
        
        # Construct the prompt
        # For chat-optimized models
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to simple concatenation
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n\n"
            prompt += "Assistant: "
        
        # Generate the response
        generated_text = ""
        
        for token in generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=max_tokens, 
            temp=temperature,
            stream=True
        ):
            socketio.emit('message', {'text': token, 'session_id': session_id})
            generated_text += token
            time.sleep(0.01)  # Small delay to avoid overwhelming the client
        
        # Signal completion
        socketio.emit('done', {'session_id': session_id})
        
        # Add the response to the chat history
        if session_id in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[session_id].add_assistant_message(generated_text.strip())
    
    except Exception as e:
        socketio.emit('error', {'error': str(e), 'session_id': session_id})


# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template(
        'index.html', 
        models=MODELS,
        default_model=DEFAULT_MODEL
    )


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return the list of available models."""
    return jsonify({
        'models': MODELS,
        'flash_attention_available': FLASH_ATTENTION_AVAILABLE,
        'flash_attention_enabled': FLASH_ATTENTION_ENABLED
    })


@app.route('/api/create_session', methods=['POST'])
def create_session():
    """Create a new chat session."""
    data = request.json
    session_id = data.get('session_id', str(int(time.time())))
    system_message = data.get('system_message', "You are a helpful assistant.")
    
    # Create a new chat session
    ACTIVE_SESSIONS[session_id] = create_chat_session(system_message=system_message)
    
    return jsonify({'session_id': session_id})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a chat message."""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    model_path = data.get('model', DEFAULT_MODEL)
    temperature = float(data.get('temperature', 0.7))
    max_tokens = int(data.get('max_tokens', 1024))
    
    # Check if session exists
    if session_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id] = create_chat_session()
    
    # Add the user message to the chat history
    ACTIVE_SESSIONS[session_id].add_user_message(message)
    
    # Get the formatted messages
    formatted_messages = ACTIVE_SESSIONS[session_id].get_formatted_context("mlx")
    
    # Start a thread to run MLX and stream the output
    threading.Thread(
        target=generate_mlx_response_stream,
        args=(model_path, formatted_messages, session_id, max_tokens, temperature)
    ).start()
    
    return jsonify({'status': 'streaming'})


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get the chat history for a session."""
    session_id = request.args.get('session_id')
    
    if session_id not in ACTIVE_SESSIONS:
        return jsonify({'error': 'Session not found'}), 404
    
    history = ACTIVE_SESSIONS[session_id].history
    
    return jsonify({
        'messages': [msg.to_dict() for msg in history.messages]
    })


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear the chat history for a session."""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in ACTIVE_SESSIONS:
        return jsonify({'error': 'Session not found'}), 404
    
    ACTIVE_SESSIONS[session_id].clear_history()
    
    return jsonify({'status': 'success'})


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('Client disconnected')


def main():
    """Main function for the web UI."""
    global MODELS, DEFAULT_MODEL, FLASH_ATTENTION_ENABLED
    
    parser = argparse.ArgumentParser(description="Web UI for MLX Chat Interface")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5001,
        help="Port to run the web server on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host to run the web server on"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--disable-flash-attention", 
        action="store_true",
        help="Disable Flash Attention optimization"
    )
    parser.add_argument(
        "--benchmark-attention", 
        action="store_true",
        help="Run attention benchmark on server start"
    )
    
    args = parser.parse_args()
    
    # Handle Flash Attention settings
    if args.disable_flash_attention:
        FLASH_ATTENTION_ENABLED = False
    
    # Check if MLX is installed
    if not check_mlx_installation():
        print("Error: MLX is not installed.")
        print("Please install MLX with: pip install mlx mlx-lm")
        return 1
    
    # List available models
    MODELS = list_available_models()
    if not MODELS:
        print("Warning: No models found in the models directory.")
        print("You will need to specify a model path manually.")
    else:
        DEFAULT_MODEL = MODELS[0]
    
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Create the HTML template
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLX Chat Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/socket.io@4.4.1/client-dist/socket.io.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .message-user {
            background-color: #f0f4f8;
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .message-assistant {
            background-color: #f0f7ff;
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .message-system {
            background-color: #fff8e6;
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        #chat-container {
            height: calc(100vh - 11rem);
            overflow-y: auto;
        }
        .typing-indicator::after {
            content: '...';
            animation: typing 1s infinite;
        }
        @keyframes typing {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-6 max-w-4xl">
        <header class="mb-6">
            <h1 class="text-3xl font-bold text-center text-purple-700">MLX Chat Interface</h1>
            <p class="text-center text-gray-600">Powered by MLX on Apple Silicon</p>
        </header>
        
        <div class="bg-white rounded-lg shadow-lg p-4 mb-4">
            <div class="flex flex-wrap mb-4">
                <div class="w-full md:w-1/2 px-2 mb-2">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="model-select">
                        Model
                    </label>
                    <select id="model-select" class="shadow border rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
                        {% for model in models %}
                        <option value="{{ model }}" {% if model == default_model %}selected{% endif %}>{{ model.split('/')[-1] }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="w-full md:w-1/4 px-2 mb-2">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="temperature">
                        Temperature
                    </label>
                    <input type="number" id="temperature" min="0" max="2" step="0.1" value="0.7" class="shadow border rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
                </div>
                <div class="w-full md:w-1/4 px-2 mb-2">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="max-tokens">
                        Max Tokens
                    </label>
                    <input type="number" id="max-tokens" min="1" max="4096" value="1024" class="shadow border rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
                </div>
            </div>
            
            <div class="mb-4">
                <label class="block text-gray-700 text-sm font-bold mb-2" for="system-message">
                    System Message
                </label>
                <textarea id="system-message" rows="2" class="shadow border rounded w-full py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">You are a helpful assistant.</textarea>
            </div>
            
            <div class="flex justify-end mb-4">
                <button id="clear-btn" class="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
                    Clear Chat
                </button>
            </div>
        </div>
        
        <div id="chat-container" class="bg-white rounded-lg shadow-lg p-4 mb-4">
            <div id="chat-messages"></div>
            <div id="typing-indicator" class="hidden">
                <div class="message-assistant typing-indicator">
                    Assistant is typing
                </div>
            </div>
        </div>
        
        <div class="flex">
            <input id="user-input" type="text" placeholder="Type your message here..." class="shadow border rounded-l w-full py-3 px-4 text-gray-700 focus:outline-none focus:shadow-outline">
            <button id="send-btn" class="bg-purple-500 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-r focus:outline-none focus:shadow-outline">
                Send
            </button>
        </div>
    </div>
    
    <script>
        // Initialize variables
        let sessionId = null;
        let isGenerating = false;
        let socket = io();
        let currentResponse = '';
        
        // DOM elements
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const modelSelect = document.getElementById('model-select');
        const temperatureInput = document.getElementById('temperature');
        const maxTokensInput = document.getElementById('max-tokens');
        const systemMessageInput = document.getElementById('system-message');
        const typingIndicator = document.getElementById('typing-indicator');
        const chatContainer = document.getElementById('chat-container');
        
        // Create a session on page load
        async function createSession() {
            sessionId = Date.now().toString();
            const response = await fetch('/api/create_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    system_message: systemMessageInput.value
                })
            });
            
            const data = await response.json();
            sessionId = data.session_id;
            
            // Add system message to the chat
            addMessage('system', systemMessageInput.value);
        }
        
        // Handle sending a message
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isGenerating) return;
            
            // Add the message to the chat
            addMessage('user', message);
            
            // Clear the input
            userInput.value = '';
            
            // Show typing indicator
            isGenerating = true;
            currentResponse = '';
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            
            // Create a response container
            const responseEl = document.createElement('div');
            responseEl.id = 'current-response';
            responseEl.className = 'message-assistant';
            
            // Add the role label
            const roleLabel = document.createElement('div');
            roleLabel.className = 'font-bold text-sm';
            roleLabel.textContent = 'Assistant';
            responseEl.appendChild(roleLabel);
            
            // Add the content
            const contentEl = document.createElement('div');
            contentEl.className = 'whitespace-pre-wrap';
            contentEl.id = 'response-content';
            contentEl.textContent = '';
            responseEl.appendChild(contentEl);
            
            chatMessages.appendChild(responseEl);
            
            // Send the message to the server
            await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: message,
                    model: modelSelect.value,
                    temperature: parseFloat(temperatureInput.value),
                    max_tokens: parseInt(maxTokensInput.value)
                })
            });
        }
        
        // Add a message to the chat
        function addMessage(role, content) {
            const messageEl = document.createElement('div');
            messageEl.className = 'message-' + role;
            
            // Add the role label
            const roleLabel = document.createElement('div');
            roleLabel.className = 'font-bold text-sm';
            roleLabel.textContent = role.charAt(0).toUpperCase() + role.slice(1);
            messageEl.appendChild(roleLabel);
            
            // Add the content
            const contentEl = document.createElement('div');
            contentEl.className = 'whitespace-pre-wrap';
            contentEl.textContent = content;
            messageEl.appendChild(contentEl);
            
            chatMessages.appendChild(messageEl);
            scrollToBottom();
        }
        
        // Clear the chat history
        async function clearChat() {
            if (isGenerating) return;
            
            await fetch('/api/clear_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
            
            chatMessages.innerHTML = '';
            
            // Add system message
            addMessage('system', systemMessageInput.value);
        }
        
        // Scroll to the bottom of the chat
        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Socket.io event handlers
        socket.on('message', (data) => {
            if (data.session_id !== sessionId) return;
            
            // Hide typing indicator
            typingIndicator.classList.add('hidden');
            
            // Update the response
            currentResponse += data.text;
            const contentEl = document.getElementById('response-content');
            if (contentEl) {
                contentEl.textContent = currentResponse;
                scrollToBottom();
            }
        });
        
        socket.on('done', (data) => {
            if (data.session_id !== sessionId) return;
            
            isGenerating = false;
            typingIndicator.classList.add('hidden');
            
            // Remove the current-response ID
            const responseEl = document.getElementById('current-response');
            if (responseEl) {
                responseEl.id = 'response-' + Date.now();
            }
        });
        
        socket.on('error', (data) => {
            if (data.session_id !== sessionId) return;
            
            console.error('Error:', data.error);
            isGenerating = false;
            typingIndicator.classList.add('hidden');
            
            // Display error message
            const errorEl = document.createElement('div');
            errorEl.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4';
            errorEl.textContent = 'Error: ' + data.error;
            chatMessages.appendChild(errorEl);
            scrollToBottom();
        });
        
        // Event listeners
        window.addEventListener('load', createSession);
        
        sendBtn.addEventListener('click', sendMessage);
        
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        clearBtn.addEventListener('click', clearChat);
        
        systemMessageInput.addEventListener('blur', async () => {
            // Update the system message
            const response = await fetch('/api/clear_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
            
            await fetch('/api/create_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    system_message: systemMessageInput.value
                })
            });
            
            // Update the displayed system message
            const systemMsgEl = document.querySelector('.message-system');
            if (systemMsgEl) {
                systemMsgEl.querySelector('.whitespace-pre-wrap').textContent = systemMessageInput.value;
            } else {
                addMessage('system', systemMessageInput.value);
            }
        });
    </script>
</body>
</html>""")
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Run attention benchmark if requested
    if args.benchmark_attention and FLASH_ATTENTION_AVAILABLE:
        print("\n🔬 Running Flash Attention benchmark...")
        try:
            benchmark = FlashAttentionBenchmark()
            benchmark.benchmark_attention_performance(
                batch_sizes=[1, 2],
                seq_lengths=[64, 128],
                head_dims=[32, 64],
                num_heads=8,
                num_runs=3
            )
            benchmark.print_summary()
        except Exception as e:
            print(f"⚠️ Benchmark failed: {e}")
    
    # Print startup message
    print(f"Starting MLX web chat server on http://{args.host}:{args.port}")
    print(f"⚡ Flash Attention: {'✅ Enabled' if FLASH_ATTENTION_ENABLED else '❌ Disabled'}")
    if MODELS:
        print(f"Found {len(MODELS)} models:")
        for model in MODELS:
            print(f"  - {os.path.basename(model)}")
    else:
        print("No models found. You will need to specify a model path manually.")
    
    # Start the server
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())