#!/bin/bash

# Basic setup demo
# Shows how to use auto-browser for simple data extraction

# Extract stock price
echo "Extracting stock price for AAPL..."
auto-browser easy -v "https://www.google.com/finance" "Get AAPL stock price"

# Extract with report
echo -e "\nExtracting with detailed report..."
auto-browser easy -v -r "https://www.google.com/finance" "Get AAPL stock price and market cap"

# Using a template
echo -e "\nCreating template for Google Finance..."
auto-browser create-template "https://www.google.com/finance" --name finance --description "Extract stock information"

echo -e "\nUsing template to extract data..."
auto-browser easy --site finance "https://www.google.com/finance" "Get TSLA stock price"
