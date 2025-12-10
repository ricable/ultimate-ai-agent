#!/bin/bash

# Movie reviews demo
# Shows how to extract and analyze movie reviews

# Create template for movie reviews
echo "Creating template for movie reviews..."
auto-browser create-template "https://www.rottentomatoes.com" --name movies --description "Extract movie reviews and ratings"

# Get current movie reviews
echo -e "\nGetting latest movie reviews..."
auto-browser easy -v -r --site movies "https://www.rottentomatoes.com" "Get reviews and ratings for the latest releases"

# Get specific movie details
echo -e "\nGetting specific movie details..."
auto-browser easy -v -r --site movies "https://www.rottentomatoes.com" "Find Oppenheimer and extract its critic score, audience score, and top reviews"

# Compare movies
echo -e "\nComparing movies..."
auto-browser easy -v -r --site movies "https://www.rottentomatoes.com" "Compare ratings and reviews for Barbie and Oppenheimer"
