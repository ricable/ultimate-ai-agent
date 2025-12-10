#!/bin/bash

# Form interaction demo
# Shows how to interact with forms and search

# Search and interact
echo "Searching and interacting..."
auto-browser easy --interactive -v "https://www.google.com/finance" "Search for AMD stock and click on the news tab"

# Multiple interactions
echo -e "\nMultiple interactions..."
auto-browser easy --interactive -v "https://www.google.com/finance" "Search for INTC, click on charts, then get the 1-year price change"
