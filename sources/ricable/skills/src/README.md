# Ericsson RAN Features Integration Pipeline

Complete pipeline for converting Ericsson technical documentation into Claude AI skills. This system processes local markdown files containing feature documentation and creates comprehensive, searchable knowledge bases optimized for Claude.

## Quick Start

### 1. Install Dependencies
```bash
cd src
pip3 install -r requirements.txt
```

### 2. Test with 5 Files (Recommended First Step)
```bash
python3 ericsson_integration.py --test
```

### 3. Process All Files
```bash
python3 ericsson_integration.py --source ../elex_features_only
```

### 4. Upload to Claude
Find the generated `.zip` file in the `output/` directory and upload it to Claude.ai/skills.

## Features

- **Scalable Processing**: Handles 5 to 2000+ files with batch processing
- **Complete Feature Extraction**: FAJ IDs, CXC codes, parameters, counters, dependencies
- **Smart Caching**: Avoids reprocessing unchanged files
- **Error Recovery**: Graceful handling of malformed files with detailed logging
- **Professional CLI**: Comprehensive command-line interface with progress reporting
- **Validation**: Built-in validation of generated skill packages

## Command Line Options

```bash
python3 ericsson_integration.py [OPTIONS]

Options:
  --source DIR      Source directory with markdown files (default: elex_features_only)
  --output DIR      Output directory (default: output)
  --test            Run in test mode with 5 files
  --limit N         Limit processing to N files
  --batch-size N    Batch size for processing (default: 50)
  --resume          Resume from cached data if available
  --help            Show help message
```

## Usage Examples

### Test Mode (5 files)
```bash
python3 ericsson_integration.py --test
```

### Custom Source Directory
```bash
python3 ericsson_integration.py --source /path/to/ericsson/docs
```

### Limited Processing
```bash
python3 ericsson_integration.py --limit 100 --batch-size 25
```

### Resume from Cache
```bash
python3 ericsson_integration.py --resume
```

## Output Structure

```
output/
├── ericsson_data/              # Processed feature data
│   ├── features/              # Individual feature JSON files
│   ├── indices/               # Search indices
│   ├── cache/                 # File-based cache
│   └── summary.json           # Processing summary
├── ericsson/                  # Generated Claude skill
│   ├── SKILL.md               # Main skill description
│   └── references/            # Reference documentation
└── ericsson_ran_features_skill_XXXX_features.zip  # Ready to upload
```

## Generated Skill Capabilities

Once uploaded to Claude, the skill enables:

### Feature Information
- "Tell me about FAJ 121 3094"
- "Show all MIMO features"
- "Which features use parameter X?"

### Technical Details
- "What does MimoSleepFunction do?"
- "Explain pmMimoSleepTime counter"
- "What is the network impact of MIMO Sleep Mode?"

### Activation and Configuration
- "How do I activate CXC4011808?"
- "What are the prerequisites for feature Y?"
- "How should I configure MIMO Sleep Mode?"

## Error Handling

The system includes comprehensive error handling:

- **Graceful Failure**: Skips malformed files but continues processing
- **Detailed Logging**: All errors are logged with file context
- **Partial Output**: Available even with some failures
- **Resume Capability**: Can resume from cached data

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| 5-file test | ~30 seconds | Validation and sample generation |
| 100 files | ~2-3 minutes | With caching |
| 2000+ files | ~15-30 minutes | Scalable batch processing |
| Skill generation | ~2-5 minutes | Reference creation and packaging |

## Troubleshooting

### No Files Found
```bash
# Check source directory
ls elex_features_only/
find elex_features_only/ -name "*.md" | wc -l
```

### No Features Extracted
- Verify markdown files contain FAJ IDs in format "FAJ XXX XXXX"
- Check file encoding (should be UTF-8)
- Ensure files are not empty or corrupted

### Memory Issues
Reduce batch size:
```bash
python3 ericsson_integration.py --batch-size 25
```

### Cache Issues
Clear cache and reprocess:
```bash
rm -rf output/ericsson_data/cache/
python3 ericsson_integration.py --test
```

## Architecture

The system consists of three main components:

1. **EricssonFeatureProcessor** - Core markdown processing engine
2. **EricssonSkillGenerator** - Claude skill structure creation
3. **EricssonIntegration** - Pipeline orchestration and CLI

Each component is modular and can be used independently if needed.

## Data Model

Extracted features include:
- Identity (FAJ ID, name, CXC code)
- Classification (value package, access type, node type)
- Technical details (parameters, counters, events)
- Dependencies and relationships
- Activation/deactivation commands
- Engineering guidelines
- Performance and network impact

## Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional feature extraction patterns
- Custom categorization rules
- Enhanced error recovery
- Performance optimizations