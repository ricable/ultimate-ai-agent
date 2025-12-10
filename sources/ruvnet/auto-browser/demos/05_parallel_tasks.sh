#!/bin/bash

# Demo 5: Parallel Task Execution
# This script demonstrates running multiple browser automation tasks in parallel

# Ensure we have a config file
if [ ! -f "config.yaml" ]; then
    echo "Creating config file first..."
    ../auto-browser init config.yaml
fi

# Create a template for parallel tasks
cat > parallel_template.yaml << EOL
sites:
  parallel_demo:
    name: "Parallel Tasks Demo"
    description: "Execute multiple tasks in parallel"
    url_pattern: "https://{domain}"
    selectors:
      main_content:
        css: "main"
        description: "Main content area"
      page_title:
        css: "h1"
        description: "Page title"
      article_text:
        css: "article p"
        description: "Article paragraphs"
        multiple: true
EOL

# Add the template to config
cat parallel_template.yaml >> config.yaml

# Create a list of URLs to process in parallel
cat > urls.txt << EOL
https://example.com/page1
https://example.com/page2
https://example.com/page3
https://example.com/page4
EOL

echo "Demonstrating parallel task execution..."
echo "This will:"
echo "- Process multiple URLs simultaneously"
echo "- Extract data from different sources in parallel"
echo "- Show efficient resource utilization"
echo "- Demonstrate concurrent browser control"

../auto-browser batch urls.txt \
    --site parallel_demo \
    --continue-on-error \
    --output parallel_results \
    --verbose

echo -e "\nParallel task demonstration complete!"
echo "This shows how to efficiently process multiple tasks simultaneously."
echo -e "\nTry the next demo (06_clinical_trials.sh) to see clinical trials data extraction!"

# Clean up
rm parallel_template.yaml urls.txt
