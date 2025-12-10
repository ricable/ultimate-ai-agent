# PRP Viewer Testing Tools

This directory contains testing tools for the Archon PRP Viewer to ensure consistent rendering between the Milkdown editor view and the PRPViewer (beautiful view).

## PRP Viewer Test Tool

### Purpose

The `prp_viewer_test.py` script identifies rendering inconsistencies between different document views in the Archon UI. It helps diagnose issues like:

- Missing sections in one view but not the other
- Image placeholder rendering problems
- JSON artifacts appearing as raw text
- Content length mismatches
- Format handling differences between markdown strings and structured PRP objects

### Prerequisites

1. **Environment Setup**

   ```bash
   # Ensure you have the required environment variables
   cp .env.example .env
   # Edit .env to include:
   # VITE_SUPABASE_URL=your_supabase_url
   # VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

2. **Start Archon UI Server**

   ```bash
   cd archon-ui-main
   npm run dev
   # Server should be running on http://localhost:3737
   ```

3. **Python Dependencies**
   The script uses Playwright (already installed via crawl4ai) and other dependencies from the server requirements.

### Usage

There are two ways to run the test:

#### Option 1: From Host Machine (Recommended)

```bash
# From the project root directory
python run_prp_viewer_test.py <PROJECT_UUID>

# Example with the template showcase project
python run_prp_viewer_test.py b4cebbce-6a2c-48c8-9583-050ddf3fb9e3
```

#### Option 2: From Inside Docker Container

```bash
# Run from inside the Archon-Server container
source .env && docker exec -e SUPABASE_URL="$SUPABASE_URL" -e SUPABASE_SERVICE_KEY="$SUPABASE_SERVICE_KEY" -e ARCHON_UI_PORT="$ARCHON_UI_PORT" Archon-Server python /app/src/server/testing/prp_viewer_test.py --project-id <PROJECT_UUID>

# Copy results back to host
docker cp Archon-Server:/app/test_results ./test_results_docker
```

**Note:** Running from the host machine is recommended as it has better access to the UI server and can take screenshots in non-headless mode for debugging.

### Output

The tool generates several output files:

1. **ViewInconsistencies*{project_id}*{timestamp}.json**
   - Detailed JSON report of all issues found
   - Includes document metadata, specific issues, and screenshots paths

2. **Summary*{project_id}*{timestamp}.txt**
   - Human-readable summary of test results
   - Lists common issues and breakdown by document

3. **Screenshots**
   - `markdown_{timestamp}.png` - Captures of the Milkdown editor view
   - `beautiful_{timestamp}.png` - Captures of the PRPViewer view

### Understanding the Results

The JSON output includes:

```json
{
  "project_id": "uuid",
  "test_date": "ISO timestamp",
  "documents": [
    {
      "doc_id": "doc_id",
      "title": "Document Title",
      "type": "prp|technical|business",
      "issues": [
        {
          "type": "missing_section|image_placeholder|json_artifact|etc",
          "description": "Details about the issue"
          // Additional issue-specific fields
        }
      ]
    }
  ],
  "summary": {
    "total_documents": 5,
    "documents_with_issues": 3,
    "common_issues": ["image_placeholders", "missing_sections"],
    "issue_breakdown": {
      "missing_section": 4,
      "image_placeholder": 3,
      "json_artifact": 2
    }
  }
}
```

### Common Issues and Fixes

1. **Image Placeholders**
   - Issue: `[Image #1]` not rendering properly
   - Fix: Ensure proper markdown conversion in `processContent` function

2. **Missing Sections**
   - Issue: Sections visible in markdown but not in beautiful view
   - Fix: Add section handlers in PRPViewer component

3. **JSON Artifacts**
   - Issue: Raw JSON displayed instead of formatted content
   - Fix: Improve content type detection and formatting

4. **Content Structure Mismatch**
   - Issue: Documents stored as both strings and objects
   - Fix: Normalize document structure before rendering

### Next Steps

After running the test tool:

1. Review the generated report to identify patterns
2. Fix the most common issues first
3. Re-run tests to verify fixes
4. Consider adding automated tests to CI/CD pipeline

### Troubleshooting

- **"Cannot connect to Archon UI server"**: Ensure the UI dev server is running on port 3737
- **"Missing Supabase credentials"**: Check your .env file has the required variables
- **"No documents found"**: Verify the project ID exists and has documents
- **Browser not launching**: Try setting `headless=True` in the script for server environments
