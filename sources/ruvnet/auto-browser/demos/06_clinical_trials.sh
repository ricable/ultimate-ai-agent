#!/bin/bash

# Clinical trials demo
# Shows how to extract structured data from clinical trials

# Create template for clinical trials
echo "Creating template for clinical trials..."
auto-browser create-template "https://clinicaltrials.gov/search" --name trials --description "Extract clinical trial information"

# Search for specific trials
echo -e "\nSearching for COVID-19 trials..."
auto-browser easy -v -r --site trials "https://clinicaltrials.gov/search" "Find phase 3 COVID-19 trials and extract their status, locations, and enrollment numbers"

# Get detailed trial information
echo -e "\nGetting detailed trial information..."
auto-browser easy -v -r --site trials "https://clinicaltrials.gov/search" "Find the latest cancer immunotherapy trials and extract their eligibility criteria and primary outcomes"
