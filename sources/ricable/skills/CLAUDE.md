# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains two main components:

1. **Skill Seeker** - Automatically converts any documentation website into a Claude AI skill. It scrapes documentation, organizes content, extracts code patterns, and packages everything into an uploadable `.zip` file for Claude.

2. **Ericsson RAN Features Processor** - A specialized system for processing Ericsson technical feature documentation from local markdown files into comprehensive Claude skills. Handles FAJ numbers, CXC codes, parameters, counters, and engineering guidelines.

## Prerequisites

**Python Version:** Python 3.7 or higher

**Required Dependencies:**
```bash
pip3 install requests beautifulsoup4 markdown python-dataclasses
```

**Optional (for API-based enhancement):**
```bash
pip3 install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Core Commands

### Ericsson RAN Features Processing

**Test with 5 Files (Recommended First Step):**
```bash
python3 src/test_ericsson_processor.py
```

**Process All Ericsson Features:**
```bash
# 1. Process markdown files into structured data
python3 src/ericsson_feature_processor.py --source elex_features_only --output output

# 2. Generate Claude skill from processed data
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data --output-dir output
```

**Limited Processing (for testing):**
```bash
# Process only 20 files
python3 src/ericsson_feature_processor.py --source elex_features_only --limit 20
```

### Web Documentation Scraping (Skill Seeker)

**Quick Start - Use a Preset:**
```bash
# Scrape and build with a preset configuration
python3 doc_scraper.py --config configs/godot.json
python3 doc_scraper.py --config configs/react.json
python3 doc_scraper.py --config configs/vue.json
python3 doc_scraper.py --config configs/django.json
python3 doc_scraper.py --config configs/fastapi.json
```

### First-Time User Workflow (Recommended)

#### For Ericsson RAN Features:

```bash
# 1. Install dependencies (one-time)
pip3 install -r src/requirements.txt

# 2. Test with 5 files to validate setup
python3 src/test_ericsson_processor.py
# Time: ~30 seconds, processes 5 files and creates sample skill

# 3. Process all documentation (if test succeeds)
python3 src/ericsson_feature_processor.py --source elex_features_only
# Time: 15-30 minutes for 2000+ files

# 4. Generate Claude skill
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data
# Time: 2-5 minutes

# Result: ericsson_ran_features_skill_XXXX_features.zip ready to upload to Claude
```

#### For Web Documentation (Skill Seeker):

```bash
# 1. Install dependencies (one-time)
pip3 install requests beautifulsoup4

# 2. Estimate page count BEFORE scraping (fast, no data download)
python3 estimate_pages.py configs/godot.json
# Time: ~1-2 minutes, shows estimated total pages and recommended max_pages

# 3. Scrape with local enhancement (uses Claude Code Max, no API key)
python3 doc_scraper.py --config configs/godot.json --enhance-local
# Time: 20-40 minutes scraping + 60 seconds enhancement

# 4. Package the skill
python3 package_skill.py output/godot/

# Result: godot.zip ready to upload to Claude
```

### Interactive Mode

```bash
# Step-by-step configuration wizard
python3 doc_scraper.py --interactive
```

### Quick Mode (Minimal Config)

```bash
# Create skill from any documentation URL
python3 doc_scraper.py --name react --url https://react.dev/ --description "React framework for UIs"
```

### Skip Scraping (Use Cached Data)

```bash
# Fast rebuild using previously scraped data
python3 doc_scraper.py --config configs/godot.json --skip-scrape
# Time: 1-3 minutes (instant rebuild)
```

### Enhancement Options

**LOCAL Enhancement (Recommended - No API Key Required):**
```bash
# During scraping
python3 doc_scraper.py --config configs/react.json --enhance-local

# Standalone after scraping
python3 enhance_skill_local.py output/react/
```

**API Enhancement (Alternative - Requires API Key):**
```bash
# During scraping
python3 doc_scraper.py --config configs/react.json --enhance

# Standalone after scraping
python3 enhance_skill.py output/react/
python3 enhance_skill.py output/react/ --api-key sk-ant-...
```

### Package the Skill

```bash
# Package skill directory into .zip file
python3 package_skill.py output/godot/
# Result: output/godot.zip
```

### Force Re-scrape

```bash
# Delete cached data and re-scrape from scratch
rm -rf output/godot_data/
python3 doc_scraper.py --config configs/godot.json
```

### Estimate Page Count (Before Scraping)

```bash
# Quick estimation - discover up to 100 pages
python3 estimate_pages.py configs/react.json --max-discovery 100
# Time: ~30-60 seconds

# Full estimation - discover up to 1000 pages (default)
python3 estimate_pages.py configs/godot.json
# Time: ~1-2 minutes

# Deep estimation - discover up to 2000 pages
python3 estimate_pages.py configs/vue.json --max-discovery 2000
# Time: ~3-5 minutes

# What it shows:
# - Estimated total pages
# - Recommended max_pages value
# - Estimated scraping time
# - Discovery rate (pages/sec)
```

**Why use estimation:**
- Validates config URL patterns before full scrape
- Helps set optimal `max_pages` value
- Estimates total scraping time
- Fast (only HEAD requests + minimal parsing)
- No data downloaded or stored

## Repository Architecture

### File Structure

```
skills/
├── src/                        # Ericsson RAN Features Processor
│   ├── ericsson_feature_processor.py    # Core processing engine (740 lines)
│   ├── ericsson_skill_generator.py      # Skill generation (526 lines)
│   ├── test_ericsson_processor.py       # Test script for 5 files
│   └── requirements.txt                 # Dependencies
├── Skill_Seekers/               # Web documentation scraper
│   ├── doc_scraper.py          # Main tool (single-file, ~790 lines)
│   ├── estimate_pages.py       # Page count estimator (fast, no data)
│   ├── enhance_skill.py        # AI enhancement (API-based)
│   ├── enhance_skill_local.py  # AI enhancement (LOCAL, no API)
│   ├── package_skill.py        # Skill packager
│   ├── run_tests.py            # Test runner (71 tests)
│   └── configs/                # Preset configurations
│       ├── godot.json
│       ├── react.json
│       ├── vue.json
│       ├── django.json
│       └── fastapi.json
├── elex_features_only/         # Ericsson documentation source (if present)
│   ├── batch_1/                # Markdown files
│   ├── batch_2/
│   └── ...
├── docs/                       # Documentation
│   ├── CLAUDE.md               # Detailed technical architecture
│   ├── ENHANCEMENT.md          # Enhancement guide
│   └── UPLOAD_GUIDE.md         # How to upload skills
├── output/                     # Generated output (git-ignored)
│   ├── ericsson_data/          # Processed Ericsson features
│   │   ├── features/*.json     # Individual feature data
│   │   ├── indices/*.json      # Search indices
│   │   └── summary.json        # Processing summary
│   └── ericsson/               # Built Ericsson skill
│       ├── SKILL.md
│       └── references/
├── plan.md                     # Original plan
├── plan_ericsson.md           # Ericsson-specific plan
├── final-plan.md              # Merged comprehensive plan
└── CLAUDE.md                  # This file
```

### Data Flow

#### Ericsson RAN Features Processor

1. **Processing Phase** (`process_all()` in src/ericsson_feature_processor.py:125-158):
   - Input: Local markdown files from `elex_features_only/`
   - Process: Batch processing (50 files per batch) → Markdown parsing → Feature extraction
   - Output: `output/ericsson_data/features/*.json` + search indices + summary.json

2. **Feature Extraction** (`process_file()` in src/ericsson_feature_processor.py:193-254):
   - Extract: FAJ numbers, CXC codes, parameters, counters, events, dependencies
   - Parse: Engineering guidelines, activation/deactivation steps
   - Cache: File-based caching for incremental updates

3. **Skill Generation Phase** (`generate_skill()` in src/ericsson_skill_generator.py:53-69):
   - Input: Processed JSON data from `output/ericsson_data/`
   - Process: Create skill structure → Generate categorized references → Create SKILL.md
   - Output: `output/ericsson/` skill directory with references

4. **Package Phase** (`package_skill()` in src/ericsson_skill_generator.py:476-501):
   - Input: Skill directory
   - Process: Zip all files (excluding .backup)
   - Output: `ericsson_ran_features_skill_XXXX_features.zip`

#### Web Documentation Scraper (Skill Seeker)

1. **Scrape Phase** (`scrape_all()` in doc_scraper.py:228-251):
   - Input: Config JSON (name, base_url, selectors, url_patterns, categories)
   - Process: BFS traversal from base_url, respecting include/exclude patterns
   - Output: `output/{name}_data/pages/*.json` + `summary.json`

2. **Build Phase** (`build_skill()` in doc_scraper.py:561-601):
   - Input: Scraped JSON data from `output/{name}_data/`
   - Process: Load pages → Smart categorize → Extract patterns → Generate references
   - Output: `output/{name}/SKILL.md` + `output/{name}/references/*.md`

3. **Enhancement Phase** (optional):
   - Input: Built skill directory with references
   - Process: Claude analyzes references and rewrites SKILL.md
   - Output: Enhanced SKILL.md with real examples and guidance

4. **Package Phase**:
   - Input: Skill directory
   - Process: Zip all files (excluding .backup)
   - Output: `{name}.zip`

### Configuration File Structure

Config files (`configs/*.json`) define scraping behavior:

```json
{
  "name": "godot",
  "description": "When to use this skill",
  "base_url": "https://docs.godotengine.org/en/stable/",
  "selectors": {
    "main_content": "div[role='main']",
    "title": "title",
    "code_blocks": "pre"
  },
  "url_patterns": {
    "include": [],
    "exclude": ["/search.html", "/_static/"]
  },
  "categories": {
    "getting_started": ["introduction", "getting_started"],
    "scripting": ["scripting", "gdscript"],
    "api": ["api", "reference", "class"]
  },
  "rate_limit": 0.5,
  "max_pages": 500
}
```

**Config Parameters:**
- `name`: Skill identifier (output directory name)
- `description`: When Claude should use this skill
- `base_url`: Starting URL for scraping
- `selectors.main_content`: CSS selector for main content (common: `article`, `main`, `div[role="main"]`)
- `selectors.title`: CSS selector for page title
- `selectors.code_blocks`: CSS selector for code samples
- `url_patterns.include`: Only scrape URLs containing these patterns
- `url_patterns.exclude`: Skip URLs containing these patterns
- `categories`: Keyword mapping for categorization
- `rate_limit`: Delay between requests (seconds)
- `max_pages`: Maximum pages to scrape

## Key Features & Implementation

### Ericsson RAN Features Processor

#### Scalable Batch Processing
- Processes files in configurable batches (default: 50 files)
- Memory-efficient for large datasets (2000+ files)
- Progress saving every 5 batches for resume capability

See: `process_batch()` in src/ericsson_feature_processor.py:160-191

#### Feature Identity Extraction
Extracts complete feature information:
- FAJ numbers (FAJ XXX XXXX format)
- CXC codes for feature activation
- Feature names and descriptions
- Value packages and node types

See: `extract_feature_identity()` in src/ericsson_feature_processor.py:256-311

#### Technical Details Extraction
Comprehensive extraction of:
- **Parameters**: From markdown tables with MO classes and descriptions
- **Counters**: PM counters with category classification
- **Events**: Event definitions with triggers and parameters
- **Dependencies**: Prerequisites, conflicts, and related features
- **Engineering Guidelines**: Best practices and configuration advice

See: `extract_parameters()`, `extract_counters()`, `extract_events()` in src/ericsson_feature_processor.py:351-458

#### Activation/Deactivation Commands
Extracts exact MO configuration steps:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011808 MO instance.
```

See: `extract_activation_step()` in src/ericsson_feature_processor.py:485-511

#### Caching System
File-based caching with MD5 hash validation:
- Avoids reprocessing unchanged files
- Enables incremental updates
- Dramatically speeds up subsequent runs

See: `process_file()` cache logic in src/ericsson_feature_processor.py:195-203

#### Search Index Generation
Creates multiple indices for fast lookup:
- Parameter index: Maps parameter names to features
- Counter index: Maps PM counters to features
- CXC index: Maps CXC codes to features
- Name index: Tokenized feature name search

See: `build_indices()` in src/ericsson_feature_processor.py:587-610

### Web Documentation Scraper (Skill Seeker)

#### Auto-Detect Existing Data
Tool checks for `output/{name}_data/` and prompts to reuse, avoiding re-scraping (check_existing_data() in doc_scraper.py:653-660).

#### Language Detection
Detects code languages from:
1. CSS class attributes (`language-*`, `lang-*`)
2. Heuristics (keywords like `def`, `const`, `func`, etc.)

See: `detect_language()` in doc_scraper.py:135-165

#### Pattern Extraction
Looks for "Example:", "Pattern:", "Usage:" markers in content and extracts following code blocks (up to 5 per page).

See: `extract_patterns()` in doc_scraper.py:167-183

#### Smart Categorization
- Scores pages against category keywords (3 points for URL match, 2 for title, 1 for content)
- Threshold of 2+ for categorization
- Auto-infers categories from URL segments if none provided
- Falls back to "other" category

See: `smart_categorize()` and `infer_categories()` in doc_scraper.py:282-351

#### Enhanced SKILL.md Generation
Generated with:
- Real code examples from documentation (language-annotated)
- Quick reference patterns extracted from docs
- Common pattern section
- Category file listings

See: `create_enhanced_skill_md()` in doc_scraper.py:426-542

## Common Workflows

### Ericsson RAN Features Processing

#### First Time (Complete Processing)

```bash
# 1. Test with 5 files to validate setup
python3 src/test_ericsson_processor.py

# 2. Process all features (if test succeeds)
python3 src/ericsson_feature_processor.py --source elex_features_only --output output

# 3. Generate Claude skill
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data --output-dir output

# Result: ericsson_ran_features_skill_XXXX_features.zip ready for Claude
# Time: 15-30 minutes (processing) + 2-5 minutes (skill generation)
```

#### Using Cached Data (Fast Iteration)

```bash
# Re-process with cache (only modified files)
python3 src/ericsson_feature_processor.py --source elex_features_only

# Quick skill regeneration
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data

# Time: 1-3 minutes (incremental processing) + 30 seconds (skill generation)
```

#### Limited Processing (For Testing)

```bash
# Process only 20 files for quick testing
python3 src/ericsson_feature_processor.py --source elex_features_only --limit 20
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data

# Time: 30-60 seconds (processing) + 10 seconds (skill generation)
```

### Web Documentation Scraping (Skill Seeker)

#### First Time (With Scraping + Enhancement)

```bash
# 1. Scrape + Build + AI Enhancement (LOCAL, no API key)
python3 doc_scraper.py --config configs/godot.json --enhance-local

# 2. Wait for enhancement terminal to close (~60 seconds)

# 3. Verify quality
cat output/godot/SKILL.md

# 4. Package
python3 package_skill.py output/godot/

# Result: godot.zip ready for Claude
# Time: 20-40 minutes (scraping) + 60 seconds (enhancement)
```

#### Using Cached Data (Fast Iteration)

```bash
# 1. Use existing data + Local Enhancement
python3 doc_scraper.py --config configs/godot.json --skip-scrape
python3 enhance_skill_local.py output/godot/

# 2. Package
python3 package_skill.py output/godot/

# Time: 1-3 minutes (build) + 60 seconds (enhancement)
```

#### Creating a New Framework Config

**Option 1: Interactive**
```bash
python3 doc_scraper.py --interactive
# Follow prompts, it creates the config for you
```

**Option 2: Copy and Modify**
```bash
# Copy a preset
cp configs/react.json configs/myframework.json

# Edit it
nano configs/myframework.json

# Test with limited pages first
# Set "max_pages": 20 in config

# Use it
python3 doc_scraper.py --config configs/myframework.json
```

## Testing & Verification

### Ericsson RAN Features Testing

#### Quick Test with 5 Files
```bash
python3 src/test_ericsson_processor.py
```

This will:
- Process 5 files from `elex_features_only/`
- Create test output in `test_output/`
- Generate a sample skill zip file
- Show processing statistics

#### Verify Ericsson Feature Extraction
```bash
# Check processed features
ls output/ericsson_data/features/

# Inspect a specific feature
cat output/ericsson_data/features/feature_121_3094.json

# Check processing summary
cat output/ericsson_data/summary.json

# Verify search indices
ls output/ericsson_data/indices/
cat output/ericsson_data/indices/parameters_index.json
```

#### Test Skill Generation
```bash
# Check generated skill structure
ls output/ericsson/references/

# Verify main skill file
cat output/ericsson/SKILL.md

# Check feature categorization
cat output/ericsson/references/features/index.md

# Verify CXC code index
cat output/ericsson/references/cxc_codes/index.md
```

#### Test with Limited Files
```bash
# Process only 20 files for quick testing
python3 src/ericsson_feature_processor.py --source elex_features_only --limit 20
python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data
```

### Web Documentation Testing

#### Finding the Right CSS Selectors

Before creating a config, test selectors with BeautifulSoup:

```python
from bs4 import BeautifulSoup
import requests

url = "https://docs.example.com/page"
soup = BeautifulSoup(requests.get(url).content, 'html.parser')

# Try different selectors
print(soup.select_one('article'))
print(soup.select_one('main'))
print(soup.select_one('div[role="main"]'))
print(soup.select_one('div.content'))

# Test code block selector
print(soup.select('pre code'))
print(soup.select('pre'))
```

#### Verify Output Quality

After building, verify the skill quality:

```bash
# Check SKILL.md has real examples
cat output/godot/SKILL.md

# Check category structure
cat output/godot/references/index.md

# List all reference files
ls output/godot/references/

# Check specific category content
cat output/godot/references/getting_started.md

# Verify code samples have language detection
grep -A 3 "```" output/godot/references/*.md | head -20
```

#### Test with Limited Pages

For faster testing, edit config to limit pages:

```json
{
  "max_pages": 20  // Test with just 20 pages
}
```

## Troubleshooting

### Ericsson RAN Features Processor

#### No Files Found
**Problem:** "Found 0 markdown files"

**Solution:** Check the source directory path:
```bash
# Verify directory exists and has .md files
ls elex_features_only/
find elex_features_only/ -name "*.md" | wc -l
```

#### No FAJ IDs Found
**Problem:** Files processed but no features extracted

**Solution:** Check markdown format for FAJ numbers:
- Should be in format "FAJ XXX XXXX" (e.g., "FAJ 121 3094")
- Check a few files manually: `grep "FAJ" elex_features_only/batch_1/*.md | head -5`

#### Memory Issues
**Problem:** Running out of memory with large datasets

**Solution:** Reduce batch size:
```bash
python3 src/ericsson_feature_processor.py --source elex_features_only --batch-size 20
```

#### Cache Issues
**Problem:** Changes not reflected after file updates

**Solution:** Clear cache and re-process:
```bash
rm -rf output/ericsson_data/cache/
python3 src/ericsson_feature_processor.py --source elex_features_only
```

### Web Documentation Scraper (Skill Seeker)

#### No Content Extracted
**Problem:** Pages scraped but content is empty

**Solution:** Check `main_content` selector in config. Try:
- `article`
- `main`
- `div[role="main"]`
- `div.content`

Use the BeautifulSoup testing approach above to find the right selector.

#### Poor Categorization
**Problem:** Pages not categorized well

**Solution:** Edit `categories` section in config with better keywords specific to the documentation structure. Check URL patterns in scraped data:

```bash
# See what URLs were scraped
cat output/godot_data/summary.json | grep url | head -20
```

#### Data Exists But Won't Use It
**Problem:** Tool won't reuse existing data

**Solution:** Force re-scrape:
```bash
rm -rf output/myframework_data/
python3 doc_scraper.py --config configs/myframework.json
```

#### Rate Limiting Issues
**Problem:** Getting rate limited or blocked by documentation server

**Solution:** Increase `rate_limit` value in config:
```json
{
  "rate_limit": 1.0  // Change from 0.5 to 1.0 seconds
}
```

#### Package Path Error
**Problem:** doc_scraper.py shows wrong package_skill.py path

**Expected output:**
```bash
python3 package_skill.py output/godot/
```

**Not:**
```bash
python3 /mnt/skills/examples/skill-creator/scripts/package_skill.py output/godot/
```

The correct command uses the local `package_skill.py` in the repository root.

## Key Code Locations

### Ericsson RAN Features Processor

- **Main processing**: `process_all()` src/ericsson_feature_processor.py:125-158
- **Batch processing**: `process_batch()` src/ericsson_feature_processor.py:160-191
- **File processing**: `process_file()` src/ericsson_feature_processor.py:193-254
- **Feature identity extraction**: `extract_feature_identity()` src/ericsson_feature_processor.py:256-311
- **Parameter extraction**: `extract_parameters()` src/ericsson_feature_processor.py:351-388
- **Counter extraction**: `extract_counters()` src/ericsson_feature_processor.py:396-420
- **Event extraction**: `extract_events()` src/ericsson_feature_processor.py:435-458
- **Activation commands**: `extract_activation_step()` src/ericsson_feature_processor.py:485-511
- **Engineering guidelines**: `extract_engineering_guidelines()` src/ericsson_feature_processor.py:523-546
- **Index building**: `build_indices()` src/ericsson_feature_processor.py:587-610
- **Skill generation**: `generate_skill()` src/ericsson_skill_generator.py:53-69
- **Reference creation**: `generate_references()` src/ericsson_skill_generator.py:195-220
- **Skill packaging**: `package_skill()` src/ericsson_skill_generator.py:476-501

### Web Documentation Scraper (Skill Seeker)

- **URL validation**: `is_valid_url()` doc_scraper.py:49-64
- **Content extraction**: `extract_content()` doc_scraper.py:66-133
- **Language detection**: `detect_language()` doc_scraper.py:135-165
- **Pattern extraction**: `extract_patterns()` doc_scraper.py:167-183
- **Smart categorization**: `smart_categorize()` doc_scraper.py:282-323
- **Category inference**: `infer_categories()` doc_scraper.py:325-351
- **Quick reference generation**: `generate_quick_reference()` doc_scraper.py:353-372
- **SKILL.md generation**: `create_enhanced_skill_md()` doc_scraper.py:426-542
- **Scraping loop**: `scrape_all()` doc_scraper.py:228-251
- **Main workflow**: `main()` doc_scraper.py:663-789

## Enhancement Details

### LOCAL Enhancement (Recommended)
- Uses your Claude Code Max plan (no API costs)
- Opens new terminal with Claude Code
- Analyzes reference files automatically
- Takes 30-60 seconds
- Quality: 9/10 (comparable to API version)
- Backs up original SKILL.md to SKILL.md.backup

### API Enhancement (Alternative)
- Uses Anthropic API (~$0.15-$0.30 per skill)
- Requires ANTHROPIC_API_KEY
- Same quality as LOCAL
- Faster (no terminal launch)
- Better for automation/CI

**What Enhancement Does:**
1. Reads reference documentation files
2. Analyzes content with Claude
3. Extracts 5-10 best code examples
4. Creates comprehensive quick reference
5. Adds domain-specific key concepts
6. Provides navigation guidance for different skill levels
7. Transforms 75-line templates into 500+ line comprehensive guides

## Performance

### Ericsson RAN Features Processor

| Task | Time | Notes |
|------|------|-------|
| Test with 5 files | ~30 sec | Validation and sample generation |
| Processing (2000 files) | 15-30 min | Scalable batch processing |
| Incremental processing | 1-3 min | With cache (only modified files) |
| Skill generation | 2-5 min | Reference creation and packaging |
| Limited processing (20 files) | 30-60 sec | Quick testing |

### Web Documentation Scraper (Skill Seeker)

| Task | Time | Notes |
|------|------|-------|
| Scraping | 15-45 min | First time only |
| Building | 1-3 min | Fast! |
| Re-building | <1 min | With --skip-scrape |
| Enhancement (LOCAL) | 30-60 sec | Uses Claude Code Max |
| Enhancement (API) | 20-40 sec | Requires API key |
| Packaging | 5-10 sec | Final zip |

## Additional Documentation

- **[README.md](README.md)** - Complete user documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 3 steps
- **[docs/CLAUDE.md](docs/CLAUDE.md)** - Detailed technical architecture
- **[docs/ENHANCEMENT.md](docs/ENHANCEMENT.md)** - AI enhancement guide
- **[docs/UPLOAD_GUIDE.md](docs/UPLOAD_GUIDE.md)** - How to upload skills to Claude
- **[STRUCTURE.md](STRUCTURE.md)** - Repository structure

## Notes for Claude Code

This repository contains two main systems:

### Ericsson RAN Features Processor
- Python-based system for processing Ericsson technical documentation
- Modular design with separate processing and generation modules
- Scalable batch processing for large datasets (2000+ files)
- Advanced extraction: FAJ numbers, CXC codes, parameters, counters, events
- File-based caching with MD5 validation for incremental updates
- Comprehensive search indices for fast feature lookup
- All processed data stored in `output/ericsson_data/` (git-ignored)

### Web Documentation Scraper (Skill Seeker)
- Single-file design (`doc_scraper.py` ~790 lines)
- Web-based documentation scraping with configurable selectors
- No build system, minimal dependencies
- Output is cached and reusable
- Enhancement is optional but highly recommended
- All scraped data stored in `output/` (git-ignored)

Both systems generate Claude skills with comprehensive reference documentation and can be used independently or together for different documentation sources.
