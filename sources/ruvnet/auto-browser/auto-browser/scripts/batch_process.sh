#!/bin/bash

# Example script for batch processing URLs using auto-browser

# Initialize config if it doesn't exist
if [ ! -f "config.yaml" ]; then
    ./auto-browser init config.yaml
fi

# Create URLs file if it doesn't exist
if [ ! -f "urls.txt" ]; then
    cat > urls.txt << EOL
https://en.wikipedia.org/wiki/Python_(programming_language)
https://en.wikipedia.org/wiki/Web_scraping
https://en.wikipedia.org/wiki/Command-line_interface
https://euclinicaltrials.eu/ctis-public/view/2022-500814-24-00
https://euclinicaltrials.eu/ctis-public/view/2023-509462-38-00
EOL
    echo "Created urls.txt with example URLs"
fi

# Create output directory
mkdir -p output/batch

# Process URLs based on their pattern
echo "Processing Wikipedia articles..."
grep "wikipedia" urls.txt > wiki_urls.txt
if [ -s wiki_urls.txt ]; then
    ./auto-browser batch wiki_urls.txt --site wiki --continue-on-error

    # Move outputs to dedicated folder
    mv output/*.json output/batch/ 2>/dev/null
fi

echo "Processing clinical trials..."
grep "euclinicaltrials" urls.txt > trials_urls.txt
if [ -s trials_urls.txt ]; then
    ./auto-browser batch trials_urls.txt --site clinical_trials --continue-on-error

    # Move outputs to dedicated folder
    mv output/*.json output/batch/ 2>/dev/null
fi

# Cleanup temporary files
rm -f wiki_urls.txt trials_urls.txt

echo "Done! Check output/batch/ for results"
