# Quick Start for Agents - Post Your Report

## 🚀 One Command to Post Your Report

After you finish your research and generate your report, run this:

```bash
node scripts/post-report.js $JOB_ID output/reports/markdown/${AGENT}_${JOB_ID}.md
```

## 📋 Example

```bash
# Your job ID: abc-123-def-456
# Your agent name: researcher
# Your report file: output/reports/markdown/researcher_abc-123-def-456.md

node scripts/post-report.js abc-123-def-456 output/reports/markdown/researcher_abc-123-def-456.md
```

## ✅ Success Output

```
✅ Report posted successfully!

   Job ID:       abc-123-def-456
   Report Path:  output/reports/markdown/researcher_abc-123-def-456.md
   Format:       markdown
   Size:         7526 characters
   Updated:      2025-11-04T03:30:00.000Z

   Database:     ./data/research-jobs.db
```

## ❓ Why Do This?

The database has TWO separate fields:
- `execution_log` - Contains all your stdout/stderr (logs)
- `report_content` - Contains your FINAL REPORT (needs manual post)

This script updates the `report_content` field with your clean, final report.

## 🔍 Verify It Worked

```bash
node scripts/view-job.js abc-123-def-456
```

You should see your report preview in the output!

---

**That's it!** Just one command after your research is done.
