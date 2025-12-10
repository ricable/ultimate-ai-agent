#!/bin/bash

# Example script for scraping Wikipedia articles using auto-browser

# Initialize config if it doesn't exist
if [ ! -f "config.yaml" ]; then
    ./auto-browser init config.yaml
fi

# Create output directory
mkdir -p output/wiki

# Example URLs to scrape
ARTICLES=(
    "https://en.wikipedia.org/wiki/Python_(programming_language)"
    "https://en.wikipedia.org/wiki/Web_scraping"
    "https://en.wikipedia.org/wiki/Command-line_interface"
)

# Process each article
for url in "${ARTICLES[@]}"; do
    echo "Processing: $url"
    ./auto-browser process "$url" --site wikipedia --output "output/wiki/$(basename "$url")"
done

echo "Done! Check output/wiki/ for results"
