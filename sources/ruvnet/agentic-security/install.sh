#!/bin/bash

# Colors and styling
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

# Cyberpunk ASCII art
echo -e "${CYAN}
    █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     
   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝
${MAGENTA}   ███████╗███████╗ ██████╗██╗   ██╗██████╗ ██╗████████╗██╗   ██╗
   ██╔════╝██╔════╝██╔════╝██║   ██║██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝
   ███████╗█████╗  ██║     ██║   ██║██████╔╝██║   ██║    ╚████╔╝ 
   ╚════██║██╔══╝  ██║     ██║   ██║██╔══██╗██║   ██║     ╚██╔╝  
   ███████║███████╗╚██████╗╚██████╔╝██║  ██║██║   ██║      ██║   
   ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝   ${NC}

${CYAN}[ AI-Powered Security Scanner & Auto-Fix Pipeline ]${NC}
${MAGENTA}[ Created by rUv, cause he could. ]${NC}
"

# Loading animation function
loading_animation() {
    local message="$1"
    local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    local delay=0.1
    local frame=0
    
    # Save cursor position
    echo -en "\033[s"
    
    while [ -d /proc/$! ]; do
        echo -en "\033[u\033[K${CYAN}[${frames[$frame]}]${NC} ${message}"
        frame=$(( (frame + 1) % 10 ))
        sleep $delay
    done
    
    # Clear line and show success
    echo -en "\033[u\033[K${GREEN}[✓]${NC} ${message}\n"
}

# Error handling
handle_error() {
    echo -e "${RED}[✗] Error: $1${NC}"
    exit 1
}

# Suppress output and show loading animation
execute_with_animation() {
    local command="$1"
    local message="$2"
    
    eval "$command" >/dev/null 2>&1 & loading_animation "$message"
    
    if [ $? -ne 0 ]; then
        handle_error "$message failed"
    fi
}

# Exit on error
set -e

# Install Security Tools
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Installing Security Tools                  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

# System update and dependencies
execute_with_animation "sudo apt-get update" "Updating system packages"
execute_with_animation "sudo apt-get install -y python3-pip python3-venv git curl unzip docker.io" "Installing system dependencies"

# Create virtual environment
execute_with_animation "python3 -m venv venv" "Creating virtual environment"
source venv/bin/activate

# Install security tools
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Installing Security Tools                  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

# Install Python Dependencies
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                Installing Python Dependencies               ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

execute_with_animation "pip install --upgrade pip" "Upgrading pip"
execute_with_animation "pip install -r requirements.txt" "Installing Python packages"

# Install OWASP ZAP
execute_with_animation "docker pull owasp/zap2docker-stable" "Installing OWASP ZAP"

# Install Nuclei
execute_with_animation "curl -s https://api.github.com/repos/projectdiscovery/nuclei/releases/latest | grep 'browser_download_url.*linux_amd64.zip' | cut -d '\"' -f 4 | wget -qi - && unzip -q nuclei-*-linux_amd64.zip && sudo mv nuclei /usr/local/bin/ && rm nuclei-*-linux_amd64.zip" "Installing Nuclei"

# Install Dependency-Check
execute_with_animation "wget -q https://github.com/jeremy-lin/dependency-check/releases/download/v6.5.3/dependency-check-6.5.3-release.zip && unzip -q dependency-check-6.5.3-release.zip -d dependency-check && rm dependency-check-6.5.3-release.zip" "Installing Dependency-Check"

# Install Python dependencies
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                Installing Python Dependencies               ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

execute_with_animation "pip install --upgrade pip" "Upgrading pip"
execute_with_animation "pip install -e ." "Installing package in development mode"

# Install GitHub CLI
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Installing GitHub CLI                     ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

execute_with_animation "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo \"deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install gh -y" "Installing GitHub CLI"

# Configure Docker
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Configuring Docker                        ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

if ! command -v docker &> /dev/null; then
    execute_with_animation "curl -fsSL https://get.docker.com | sh" "Installing Docker"
    execute_with_animation "sudo usermod -aG docker $USER" "Adding user to docker group"
fi

# Install and configure Docker
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Configuring Docker                        ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

if ! command -v docker &> /dev/null; then
    execute_with_animation "curl -fsSL https://get.docker.com | sh" "Installing Docker"
    execute_with_animation "sudo usermod -aG docker $USER" "Adding user to docker group"
fi

# Configure Docker
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Configuring Docker                        ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

if ! command -v docker &> /dev/null; then
    execute_with_animation "curl -fsSL https://get.docker.com | sh" "Installing Docker"
    execute_with_animation "sudo usermod -aG docker $USER" "Adding user to docker group"
fi

# Pull required Docker images
execute_with_animation "docker pull owasp/zap2docker-stable" "Pulling OWASP ZAP image"

# Configure cache directory
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                  Setting up Cache                         ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

execute_with_animation "mkdir -p .security_cache" "Creating cache directory"
execute_with_animation "chmod 755 .security_cache" "Setting cache permissions"

# Make scripts executable
execute_with_animation "chmod +x run_pipeline.sh src/agentic_security/security_cli.py" "Making scripts executable"

# Installation complete
echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}              Installation Completed Successfully           ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${CYAN}[>]${NC} To activate the virtual environment, run: ${CYAN}source venv/bin/activate${NC}"
echo -e "${CYAN}[>]${NC} Ready to hack the planet! 🌐\n"
