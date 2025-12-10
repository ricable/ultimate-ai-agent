#!/bin/bash
# Initialize and run chat interfaces

# ANSI color codes
RESET="\033[0m"
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"

# Print colorized message
print_color() {
  local color=$1
  local message=$2
  echo -e "${color}${message}${RESET}"
}

# Print header
print_color "${BOLD}${CYAN}" "=================================="
print_color "${BOLD}${CYAN}" " LLM Chat Interfaces for Mac Silicon"
print_color "${BOLD}${CYAN}" "=================================="
echo ""

# Check Python dependencies
print_color "${BLUE}" "Checking Python dependencies..."
MISSING_DEPS=0

check_dependency() {
  local package=$1
  python3 -c "import $package" 2>/dev/null
  if [ $? -ne 0 ]; then
    print_color "${YELLOW}" "- Missing package: $package"
    MISSING_DEPS=1
    return 1
  else
    print_color "${GREEN}" "- Found package: $package"
    return 0
  fi
}

check_dependency flask
check_dependency "flask_socketio"

if [ $MISSING_DEPS -eq 1 ]; then
  print_color "${YELLOW}" "Installing missing dependencies..."
  pip3 install flask flask-socketio
  if [ $? -ne 0 ]; then
    print_color "${RED}" "Failed to install dependencies. Please install them manually."
    exit 1
  fi
fi

# Create necessary directories
print_color "${BLUE}" "Ensuring directories exist..."
mkdir -p ../llama.cpp-setup/models
mkdir -p ../mlx-setup/models

# Check frameworks
print_color "${BLUE}" "Checking frameworks..."

# Check llama.cpp
if [ -x "../llama.cpp-setup/build/main" ]; then
  print_color "${GREEN}" "- llama.cpp is installed."
  LLAMACPP_INSTALLED=1
else
  print_color "${YELLOW}" "- llama.cpp executable not found. Some features may not work."
  LLAMACPP_INSTALLED=0
fi

# Check MLX
python3 -c "import mlx.core" 2>/dev/null
if [ $? -eq 0 ]; then
  print_color "${GREEN}" "- MLX is installed."
  MLX_INSTALLED=1
else
  print_color "${YELLOW}" "- MLX is not installed. Some features may not work."
  MLX_INSTALLED=0
fi

# Download test models if necessary
print_color "${BLUE}" "Checking for test models..."

download_models=0
if [ ! -d "../llama.cpp-setup/models/" ] || [ -z "$(ls -A ../llama.cpp-setup/models/)" ]; then
  print_color "${YELLOW}" "- No llama.cpp models found."
  download_models=1
fi

if [ ! -d "../mlx-setup/models/" ] || [ -z "$(ls -A ../mlx-setup/models/)" ]; then
  print_color "${YELLOW}" "- No MLX models found."
  download_models=1
fi

if [ $download_models -eq 1 ]; then
  print_color "${YELLOW}" "Would you like to download small test models? (y/n)"
  read -r download_choice
  if [[ $download_choice =~ ^[Yy]$ ]]; then
    print_color "${BLUE}" "Downloading test models..."
    python3 scripts/setup_test_model.py
  fi
fi

# Display menu
show_menu() {
  echo ""
  print_color "${BOLD}${CYAN}" "Choose an option:"
  echo ""
  print_color "${BOLD}" "Command-Line Interfaces:"
  echo "1. Run llama.cpp CLI"
  echo "2. Run MLX CLI"
  echo ""
  print_color "${BOLD}" "Web Interfaces:"
  echo "3. Run llama.cpp Web UI"
  echo "4. Run MLX Web UI"
  echo ""
  print_color "${BOLD}" "Example Workflows:"
  echo "5. Run Code Assistant Workflow"
  echo "6. Run Data Analysis Workflow"
  echo ""
  print_color "${BOLD}" "Utilities:"
  echo "7. Download Test Models"
  echo "8. View Documentation"
  echo "9. Exit"
  echo ""
  print_color "${YELLOW}" "Enter your choice (1-9):"
}

# View documentation
view_docs() {
  if command -v open >/dev/null 2>&1; then
    open "docs/improving_chat_quality.md"
  else
    print_color "${YELLOW}" "Please open the documentation manually:"
    print_color "${BLUE}" "docs/improving_chat_quality.md"
  fi
}

# Main loop
while true; do
  show_menu
  read -r choice
  
  case $choice in
    1)
      if [ $LLAMACPP_INSTALLED -eq 1 ]; then
        print_color "${GREEN}" "Starting llama.cpp CLI..."
        python3 llama_cpp/cli/chat_cli.py
      else
        print_color "${RED}" "llama.cpp is not installed. Please install it first."
      fi
      ;;
    2)
      if [ $MLX_INSTALLED -eq 1 ]; then
        print_color "${GREEN}" "Starting MLX CLI..."
        python3 mlx/cli/chat_cli.py
      else
        print_color "${RED}" "MLX is not installed. Please install it first."
      fi
      ;;
    3)
      if [ $LLAMACPP_INSTALLED -eq 1 ]; then
        print_color "${GREEN}" "Starting llama.cpp Web UI..."
        python3 llama_cpp/web/web_app.py
      else
        print_color "${RED}" "llama.cpp is not installed. Please install it first."
      fi
      ;;
    4)
      if [ $MLX_INSTALLED -eq 1 ]; then
        print_color "${GREEN}" "Starting MLX Web UI..."
        python3 mlx/web/web_app.py
      else
        print_color "${RED}" "MLX is not installed. Please install it first."
      fi
      ;;
    5)
      print_color "${GREEN}" "Running Code Assistant Workflow..."
      python3 examples/code_assistant_workflow.py
      ;;
    6)
      print_color "${GREEN}" "Running Data Analysis Workflow..."
      python3 examples/data_analysis_workflow.py
      ;;
    7)
      print_color "${GREEN}" "Downloading Test Models..."
      python3 scripts/setup_test_model.py
      ;;
    8)
      print_color "${GREEN}" "Viewing Documentation..."
      view_docs
      ;;
    9)
      print_color "${GREEN}" "Goodbye!"
      exit 0
      ;;
    *)
      print_color "${RED}" "Invalid choice. Please try again."
      ;;
  esac
  
  echo ""
  print_color "${YELLOW}" "Press Enter to continue..."
  read
done