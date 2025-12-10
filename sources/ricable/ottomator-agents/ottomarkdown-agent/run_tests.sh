#!/bin/bash

# Change to project directory
cd /Users/loic/Desktop/ottomarkdown

# Kill any existing Python processes and wait for port to be free
echo "Cleaning up existing processes..."
pkill -f "python3 file_agent.py"
lsof -ti:8001 | xargs kill -9 2>/dev/null
sleep 3

# Clean markdown_results directory
echo "Cleaning markdown_results directory..."
rm -rf markdown_results/*
mkdir -p markdown_results

# Remove .DS_Store files
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete

# Activate virtual environment
source ./venv/bin/activate

# Start the API server in the background
echo "Starting API server..."
python3 file_agent.py &
API_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
sleep 5

# Run all tests
echo "Running tests..."
python3 -c "
import validation_test
import asyncio

async def run_tests():
    validation_test.test_openrouter_api()
    validation_test.test_file_processing()
    await validation_test.test_convert_to_markdown()
    validation_test.test_file_processing_with_llm()
    validation_test.test_image_processing_with_llm()
    validation_test.test_api_file_agent_cached()

asyncio.run(run_tests())
"

# Kill the API server
echo "Cleaning up..."
kill $API_PID 2>/dev/null
pkill -f "python3 file_agent.py"

echo "Done!"
