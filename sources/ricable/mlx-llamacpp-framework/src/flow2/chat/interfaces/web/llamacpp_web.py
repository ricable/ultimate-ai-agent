#!/usr/bin/env python3
"""
Web UI for llama.cpp Chat Interface

This script provides a simple web interface for interacting with llama.cpp models.
It uses Flask for the web server and includes WebSocket support for streaming responses.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import threading
import time
from typing import List, Dict, Optional, Any, Union, Tuple

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from common.chat_history import create_chat_session, ChatHistory

# Flask and related imports
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO

# Setup Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'llamacpp-web-ui-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
MODELS = []
DEFAULT_MODEL = ""
ACTIVE_SESSIONS = {}  # Dictionary to store active chat sessions
LLAMACPP_PATH = ""


def check_llamacpp_installation() -> Tuple[bool, str]:
    """
    Check if llama.cpp is installed and return the path to the executable.
    
    Returns:
        Tuple of (is_installed, path_to_executable)
    """
    # Check the expected location first
    llamacpp_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/build/main'
    ))
    
    if os.path.exists(llamacpp_path) and os.access(llamacpp_path, os.X_OK):
        return True, llamacpp_path
    
    # Try to find using which
    try:
        result = subprocess.run(
            ['which', 'main'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
    except:
        pass
    
    return False, ""


def list_available_models() -> List[str]:
    """
    List available model files in the models directory.
    
    Returns:
        List of model paths
    """
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/models'
    ))
    
    if not os.path.exists(models_dir):
        return []
    
    return [
        os.path.join(models_dir, f) 
        for f in os.listdir(models_dir) 
        if f.endswith('.gguf')
    ]


def run_llamacpp_stream(
    model_path: str, 
    prompt: str, 
    session_id: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    use_metal: bool = True,
    prompt_template: str = "chatml"
) -> None:
    """
    Run the llama.cpp model and stream output via WebSocket.
    
    Args:
        model_path: Path to the model file
        prompt: The prompt to send to the model
        session_id: Session ID for WebSocket communication
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_metal: Whether to use Metal acceleration on macOS
        prompt_template: The prompt template type
    """
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(prompt)
        prompt_file = temp_file.name
    
    try:
        # Build the command
        cmd = [
            LLAMACPP_PATH,
            '-m', model_path,
            '-n', str(max_tokens),
            '--temp', str(temperature),
            '-f', prompt_file,
        ]
        
        if use_metal:
            cmd.extend(['--metal'])
        
        # Add template-specific parameters
        if prompt_template == "llama2":
            cmd.extend(['--format', 'llama2'])
        
        # Run llama.cpp
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Variables to track the assistant's response
        started_response = False
        assistant_response = ""
        
        # Process output
        for line in process.stdout:
            # Extract the assistant's response based on the prompt template
            if prompt_template == "chatml":
                if "<|im_start|>assistant" in line:
                    started_response = True
                    continue
                
                if started_response:
                    if "<|im_end|>" in line:
                        break
                    socketio.emit('message', {'text': line, 'session_id': session_id})
                    assistant_response += line
            
            elif prompt_template == "llama2":
                if "[/INST]" in line and not started_response:
                    parts = line.split("[/INST]", 1)
                    if len(parts) > 1:
                        started_response = True
                        socketio.emit('message', {'text': parts[1], 'session_id': session_id})
                        assistant_response += parts[1]
                        continue
                
                if started_response:
                    socketio.emit('message', {'text': line, 'session_id': session_id})
                    assistant_response += line
            
            else:
                # Simple template - just emit everything after prompt
                if not started_response and "Assistant:" in line:
                    parts = line.split("Assistant:", 1)
                    if len(parts) > 1:
                        started_response = True
                        socketio.emit('message', {'text': parts[1], 'session_id': session_id})
                        assistant_response += parts[1]
                        continue
                
                if started_response:
                    socketio.emit('message', {'text': line, 'session_id': session_id})
                    assistant_response += line
        
        # Signal completion
        socketio.emit('done', {'session_id': session_id})
        
        # Add the response to the chat history
        if session_id in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS[session_id].add_assistant_message(assistant_response.strip())
    
    except Exception as e:
        socketio.emit('error', {'error': str(e), 'session_id': session_id})
    
    finally:
        # Clean up the temporary file
        os.unlink(prompt_file)


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
    return jsonify({'models': MODELS})


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
    prompt_template = data.get('prompt_template', 'chatml')
    
    # Check if session exists
    if session_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id] = create_chat_session()
    
    # Add the user message to the chat history
    ACTIVE_SESSIONS[session_id].add_user_message(message)
    
    # Get the formatted prompt
    formatted_prompt = ACTIVE_SESSIONS[session_id].get_formatted_context(prompt_template)
    
    # Start a thread to run llama.cpp and stream the output
    threading.Thread(
        target=run_llamacpp_stream,
        args=(model_path, formatted_prompt, session_id, max_tokens, temperature, True, prompt_template)
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
    global MODELS, DEFAULT_MODEL, LLAMACPP_PATH
    
    parser = argparse.ArgumentParser(description="Web UI for llama.cpp Chat Interface")
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
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
    
    args = parser.parse_args()
    
    # Check if llama.cpp is installed
    is_installed, llamacpp_path = check_llamacpp_installation()
    if not is_installed:
        print("Error: llama.cpp executable not found.")
        print("Please ensure llama.cpp is properly installed.")
        return 1
    
    LLAMACPP_PATH = llamacpp_path
    
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
    <title>llama.cpp Chat Interface</title>
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
            background-color: #e6f7ff;
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
            <h1 class="text-3xl font-bold text-center text-blue-700">llama.cpp Chat Interface</h1>
            <p class="text-center text-gray-600">Powered by llama.cpp on Apple Silicon</p>
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
            
            <div class="flex justify-between mb-4">
                <select id="template-select" class="shadow border rounded py-2 px-3 text-gray-700 focus:outline-none focus:shadow-outline">
                    <option value="chatml">ChatML Template</option>
                    <option value="llama2">Llama2 Template</option>
                    <option value="alpaca">Alpaca Template</option>
                    <option value="simple">Simple Template</option>
                </select>
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
            <button id="send-btn" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-r focus:outline-none focus:shadow-outline">
                Send
            </button>
        </div>
    </div>
    
    <script>
        // Initialize variables
        let sessionId = null;
        let isGenerating = false;
        let socket = io();
        
        // DOM elements
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const modelSelect = document.getElementById('model-select');
        const temperatureInput = document.getElementById('temperature');
        const maxTokensInput = document.getElementById('max-tokens');
        const systemMessageInput = document.getElementById('system-message');
        const templateSelect = document.getElementById('template-select');
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
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            
            // Create a response container
            const responseId = 'response-' + Date.now();
            const responseEl = document.createElement('div');
            responseEl.id = responseId;
            responseEl.className = 'message-assistant';
            responseEl.textContent = '';
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
                    max_tokens: parseInt(maxTokensInput.value),
                    prompt_template: templateSelect.value
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
            
            const responseEl = document.getElementById('response-' + Date.now()) || 
                               document.querySelector('.message-assistant:last-child');
            
            if (responseEl) {
                responseEl.textContent += data.text;
                scrollToBottom();
            }
        });
        
        socket.on('done', (data) => {
            if (data.session_id !== sessionId) return;
            
            isGenerating = false;
            typingIndicator.classList.add('hidden');
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
    
    # Print startup message
    print(f"Starting web server on http://{args.host}:{args.port}")
    print(f"Using llama.cpp at: {LLAMACPP_PATH}")
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