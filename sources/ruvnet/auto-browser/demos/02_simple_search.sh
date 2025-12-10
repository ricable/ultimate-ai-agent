#!/bin/bash

# Simple search demo
# Shows how to search and extract data

# Search for a stock and get its price
echo "Searching for META stock..."
auto-browser easy -v -r "https://www.google.com/finance" "Search for META stock and extract its current price"

# Search and extract multiple data points
echo -e "\nGetting detailed stock information..."
auto-browser easy -v -r "https://www.google.com/finance" "Search for NVDA and get price, market cap, and latest news"
