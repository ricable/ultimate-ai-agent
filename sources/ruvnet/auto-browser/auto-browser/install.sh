#!/bin/bash

echo "Installing auto-browser..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Install Python package
pip install -e .

# Install Playwright browsers
playwright install

# Function to check and handle package manager processes
check_package_manager_processes() {
    local dpkg_lock_pid=$(sudo fuser /var/lib/dpkg/lock 2>/dev/null)
    local apt_lists_lock_pid=$(sudo fuser /var/lib/apt/lists/lock 2>/dev/null)
    local dpkg_frontend_lock_pid=$(sudo fuser /var/lib/dpkg/lock-frontend 2>/dev/null)
    
    if [ -n "$dpkg_lock_pid" ] || [ -n "$apt_lists_lock_pid" ] || [ -n "$dpkg_frontend_lock_pid" ]; then
        echo "Found package manager processes:"
        [ -n "$dpkg_lock_pid" ] && echo "- dpkg lock: PID $dpkg_lock_pid ($(ps -p $dpkg_lock_pid -o comm=))"
        [ -n "$apt_lists_lock_pid" ] && echo "- apt lists lock: PID $apt_lists_lock_pid ($(ps -p $apt_lists_lock_pid -o comm=))"
        [ -n "$dpkg_frontend_lock_pid" ] && echo "- dpkg frontend lock: PID $dpkg_frontend_lock_pid ($(ps -p $dpkg_frontend_lock_pid -o comm=))"
        
        read -p "Do you want to kill these processes? (y/N) " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            [ -n "$dpkg_lock_pid" ] && sudo kill $dpkg_lock_pid
            [ -n "$apt_lists_lock_pid" ] && sudo kill $apt_lists_lock_pid
            [ -n "$dpkg_frontend_lock_pid" ] && sudo kill $dpkg_frontend_lock_pid
            echo "Processes killed. Cleaning up lock files..."
            sudo rm -f /var/lib/dpkg/lock
            sudo rm -f /var/lib/apt/lists/lock
            sudo rm -f /var/lib/dpkg/lock-frontend
            sudo dpkg --configure -a
            return 0
        else
            echo "Continuing to wait for processes to finish..."
            return 1
        fi
    fi
    return 0
}

# Function to wait for apt locks to be released
wait_for_apt() {
    local wait_time=0
    local max_wait=60  # Maximum wait time in seconds
    
    while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1 || \
          sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || \
          sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
        
        if [ $wait_time -ge $max_wait ]; then
            echo "Package manager processes are still running after ${max_wait} seconds."
            check_package_manager_processes
            wait_time=0
        fi
        
        echo "Waiting for other package manager processes to finish... (${wait_time}s)"
        sleep 1
        ((wait_time++))
    done
}

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Install system dependencies for Playwright
echo "Installing system dependencies..."

# Wait for any existing package management processes
wait_for_apt

# Install Node.js if not present
if ! command -v npx &> /dev/null; then
    echo "Installing Node.js and npm..."
    if ! curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -; then
        handle_error "Failed to setup Node.js repository"
    fi
    wait_for_apt
    if ! sudo apt-get install -y nodejs; then
        handle_error "Failed to install Node.js"
    fi
fi

echo "Installing Playwright system dependencies..."
if ! npx playwright install-deps; then
    handle_error "Failed to install Playwright dependencies"
fi

echo "Installation complete!"
