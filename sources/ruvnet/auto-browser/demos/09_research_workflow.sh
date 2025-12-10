#!/bin/bash

# Research Workflow Demo
# Shows automated academic research and paper analysis

# Create templates
echo "Creating research templates..."
auto-browser create-template "https://scholar.google.com" --name scholar --description "Google Scholar automation"
auto-browser create-template "https://arxiv.org" --name arxiv --description "ArXiv paper extraction"
auto-browser create-template "https://zotero.org" --name zotero --description "Reference management"
auto-browser create-template "https://overleaf.com" --name overleaf --description "Paper writing"

# Initial literature search
echo -e "\nPerforming literature search..."
auto-browser easy -v -r "https://scholar.google.com" "Search for papers about 'LLM-powered automation' from last 2 years, sort by citations, extract:
- Top 20 paper titles and abstracts
- Author information and institutions
- Citation counts and links
- Related papers"

# Detailed paper analysis
echo -e "\nAnalyzing papers on ArXiv..."
auto-browser easy -v -r --site arxiv "https://arxiv.org" "For each relevant paper:
1. Download full PDF if available
2. Extract methodology section
3. List key findings and contributions
4. Note technical approaches used
5. Identify evaluation metrics"

# Reference management
echo -e "\nOrganizing references..."
auto-browser easy --interactive -v --site zotero "https://zotero.org" "Login, create new collection 'LLM Automation Research', then:
1. Import all found papers
2. Add tags for key topics
3. Extract citations
4. Download PDFs
5. Create bibliography"

# Literature analysis
echo -e "\nAnalyzing research trends..."
auto-browser easy -v -r --site scholar "https://scholar.google.com" "Analyze the papers to identify:
- Common methodologies
- Emerging trends
- Research gaps
- Key challenges
- Future directions"

# Paper writing setup
echo -e "\nSetting up paper draft..."
auto-browser easy --interactive -v --site overleaf "https://overleaf.com" "Login, create new paper:
1. Set up IEEE template
2. Create sections from template
3. Import bibliography
4. Add initial citations
5. Create figures placeholder"

# Collaboration setup
echo -e "\nSetting up collaboration..."
auto-browser easy --interactive -v --site overleaf "https://overleaf.com" "Share paper with team@university.edu, set up:
1. Edit permissions
2. Track changes
3. Comment access
4. Version control
5. Auto-compilation"

# Export and backup
echo -e "\nExporting research materials..."
auto-browser easy --interactive -v --site zotero "https://zotero.org" "Export:
1. Complete bibliography
2. PDF collection
3. Research notes
4. Citation network graph
Save to research_backup folder"
