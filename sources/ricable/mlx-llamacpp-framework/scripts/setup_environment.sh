#!/bin/bash
# Setup script for LLM on Mac Silicon

# Print colored messages
function print_blue() {
  echo -e "\033[34m$1\033[0m"
}
function print_green() {
  echo -e "\033[32m$1\033[0m"
}
function print_red() {
  echo -e "\033[31m$1\033[0m"
}
function print_yellow() {
  echo -e "\033[33m$1\033[0m"
}
function print_header() {
  echo -e "\033[1;36m==================================\033[0m"
  echo -e "\033[1;36m$1\033[0m"
  echo -e "\033[1;36m==================================\033[0m"
}

print_header "LLM on Mac Silicon Setup"

# Check for uv
UV_CMD=$(which uv)
if [ -z "$UV_CMD" ]; then
  print_yellow "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Reload shell to get uv in PATH
  export PATH="$HOME/.cargo/bin:$PATH"
  UV_CMD=$(which uv)
  if [ -z "$UV_CMD" ]; then
    print_red "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

print_green "Found uv at $UV_CMD"

# Detect Python
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
  print_red "Python 3 not found. Please install Python 3."
  exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d " " -f 2)
print_green "Found Python $PYTHON_VERSION at $PYTHON_CMD"

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
  print_yellow "Virtual environment already exists at $VENV_DIR"
  read -p "Do you want to recreate it? (y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_blue "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
  else
    print_blue "Using existing virtual environment."
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  print_blue "Creating virtual environment in $VENV_DIR with uv..."
  uv venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    print_red "Failed to create virtual environment."
    exit 1
  fi
  print_green "Virtual environment created successfully!"
fi

# Activate virtual environment
print_blue "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies using uv
print_blue "Installing dependencies with uv..."

# Basic dependencies
uv pip install numpy scipy

# MLX dependencies
print_blue "Installing MLX dependencies..."
uv pip install mlx mlx-lm

# llama.cpp dependencies
print_blue "Installing llama.cpp dependencies..."
uv pip install llama-cpp-python

# Web UI dependencies
print_blue "Installing Web UI dependencies..."
uv pip install flask flask-socketio

print_green "Dependencies installed successfully!"

# Create activation script
print_blue "Creating activation script..."
cat > activate.sh << 'EOL'
#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Virtual environment activated!"
echo "You can now run the LLM tools."
echo "Example commands:"
echo "  - python inference_scripts/mlx/inference.py --help"
echo "  - python inference_scripts/llama_cpp/inference.py --help"
echo "  - python chat_interfaces/mlx/cli/chat_cli.py --help"
EOL

chmod +x activate.sh

print_green "Setup complete!"
print_green "To activate the environment, run: source activate.sh"
print_yellow "Note: You'll need to download models before running inference."
print_yellow "See model_utils/download-model.sh for model download options."