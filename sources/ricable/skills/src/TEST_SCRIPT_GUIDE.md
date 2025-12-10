# Ericsson RAN Features Test Script Usage Guide

## Overview

The test script `src/test_ericsson_processor.py` validates the entire Ericsson RAN Features processing pipeline using just 5 files for rapid validation (~30 seconds).

## Prerequisites

1. **Dependencies**: Install required packages
   ```bash
   pip3 install --break-system-packages -r src/requirements.txt
   ```

2. **Source Files**: Ensure `elex_features_only/` directory contains markdown files
   ```bash
   find elex_features_only/ -name "*.md" | wc -l
   # Should show > 0 files (445 files in current dataset)
   ```

## Running the Test

**Basic Test:**
```bash
python3 src/test_ericsson_processor.py
```

**From Repository Root:**
```bash
python3 src/test_ericsson_processor.py
```

## Test Phases

The test script runs 5 comprehensive phases:

### Phase 0: Prerequisites Validation
- ✅ Verifies source directory exists
- ✅ Confirms sufficient markdown files available
- ✅ Cleans previous test output

### Phase 1: Processing Test
- 📊 Processes exactly 5 files from the source
- ⏱️ Tracks processing time and performance
- 🔍 Extracts features, parameters, counters, events

### Phase 2: Quality Validation
- 🔬 Analyzes extraction quality
- 📋 Reports detailed statistics per feature
- 🔧 Validates CXC codes, parameters, counters extraction

### Phase 3: Skill Generation Test
- 🚀 Generates Claude skill from processed data
- 📦 Creates downloadable zip package
- 📁 Validates skill directory structure

### Phase 4: Search Indices Validation
- 📚 Validates search indices creation
- 📊 Reports index entry counts
- ✅ Ensures all indices are properly built

### Phase 5: Comprehensive Summary
- 🎯 Displays complete test results
- ⏱️ Shows total execution time
- 🎉 Provides success/failure status
- 🚀 Lists next steps

## Expected Output

```
🧪 Ericsson RAN Features Processor - 5 File Validation Test
Based on final-plan.md requirements

🔍 Validating test prerequisites...
✅ Found 445 markdown files available

📊 Phase 1: Processing 5 test files
[Processing details...]

🔬 Phase 2: Validating extraction quality
[Feature analysis details...]

🚀 Phase 3: Generating Claude skill
[Skill generation details...]

📚 Phase 4: Validating search indices
[Index validation details...]

🎯 COMPREHENSIVE TEST SUMMARY
============================================================
📊 PROCESSING RESULTS:
  Files found: 445
  Files processed: 5
  Features extracted: 5
  Processing errors: 0
  Processing time: 0.13 seconds

🔬 EXTRACTION QUALITY:
  Features with CXC codes: X
  Features with parameters: X
  Total parameters extracted: X
  [More quality metrics...]

🚀 SKILL GENERATION:
  ✅ Generation successful
  Zip file created: ericsson_ran_features_skill_5_features.zip
  Package size: X.XX MB

🎉 ✅ TEST SUCCESSFUL!
  All validation criteria passed
  Sample skill ready for upload
```

## Test Results

### Success Criteria ✅
- **Processing**: Features extracted successfully (> 0)
- **Generation**: Skill zip file created
- **Indices**: All search indices built
- **Timing**: Test completes in ~30 seconds

### Output Files Created
```
test_output/
├── ericsson_ran_features_skill_5_features.zip    # 📦 Upload this to Claude
├── ericsson_data/                                 # 🔍 Processed data
│   ├── features/                                  # 📄 Individual feature JSON files
│   ├── indices/                                   # 📚 Search indices
│   └── summary.json                               # 📊 Processing summary
└── ericsson/                                      # 🚀 Generated skill
    ├── SKILL.md                                   # 📖 Main skill file
    └── references/                                # 📋 Reference documentation
```

## Troubleshooting

### ❌ ModuleNotFoundError
```bash
pip3 install --break-system-packages -r src/requirements.txt
```

### ❌ Source directory not found
```bash
# Verify the source directory exists
ls -la elex_features_only/
```

### ❌ No markdown files found
```bash
# Check if files exist
find elex_features_only/ -name "*.md" | wc -l
```

### ❌ No features extracted
- Check markdown file format for FAJ numbers
- Verify files contain valid Ericsson feature documentation
- Review processing error messages

## Next Steps After Successful Test

1. **Upload to Claude**:
   ```bash
   # Upload this file to Claude:
   test_output/ericsson_ran_features_skill_5_features.zip
   ```

2. **Process Full Dataset**:
   ```bash
   # Process all 445+ files
   python3 src/ericsson_feature_processor.py --source elex_features_only
   python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data
   ```

3. **Test Sample Queries**:
   - "What is MIMO Sleep Mode?"
   - "How do I configure Prescheduling?"
   - "What parameters are available for Dynamic PUCCH?"

## Performance Metrics

| Component | Typical Time | Description |
|-----------|-------------|-------------|
| Processing (5 files) | ~0.1-0.3 seconds | Feature extraction and parsing |
| Quality Validation | ~0.05 seconds | Analysis and statistics |
| Skill Generation | ~0.05 seconds | Create skill files and packaging |
| Index Validation | ~0.01 seconds | Verify search indices |
| **Total Test Time** | **~0.14 seconds** | Complete validation pipeline |

The test is designed for rapid validation and can be run multiple times during development to ensure the processing pipeline works correctly before processing the full dataset.