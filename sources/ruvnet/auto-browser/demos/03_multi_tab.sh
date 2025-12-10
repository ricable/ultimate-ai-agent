#!/bin/bash

# Multi-tab demo
# Shows how to work with multiple stocks

# Compare multiple stocks
echo "Comparing tech stocks..."
auto-browser easy -v -r "https://www.google.com/finance" "Compare AAPL, MSFT, and GOOGL stock prices"

# Extract news for multiple companies
echo -e "\nGetting latest news..."
auto-browser easy -v -r "https://www.google.com/finance" "Get latest news for AAPL and TSLA"
