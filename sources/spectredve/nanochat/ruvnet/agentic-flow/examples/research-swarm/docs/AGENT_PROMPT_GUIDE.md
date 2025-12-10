# Agent Prompt Guide - Final Report Submission

## 🎯 Purpose

This guide ensures AI agents post their final research reports correctly to the SQLite database.

## 📋 Final Step Instructions for Agents

When you complete your research task, you MUST execute this final command to post your report:

```bash
node scripts/post-report.js $JOB_ID output/reports/markdown/${AGENT}_${JOB_ID}.md
```

## ✅ Complete Agent Workflow

### 1. Research Phase
- Conduct your research using available tools
- Generate comprehensive report in markdown format
- Save report to: `output/reports/markdown/${AGENT}_${JOB_ID}.md`

### 2. Final Report Submission (CRITICAL)
```bash
# Post your final report to the database
node scripts/post-report.js $JOB_ID output/reports/markdown/${AGENT}_${JOB_ID}.md
```

### 3. Verification
```bash
# Verify your report was posted correctly
node scripts/view-job.js $JOB_ID
```

## 📝 Agent Task Template

Copy this template for your research tasks:

```markdown
**TASK**: [Your research task description]

**AGENT INSTRUCTIONS**:

1. **Research Phase**:
   - Conduct comprehensive research on the topic
   - Use MCP tools as needed
   - Apply anti-hallucination protocols
   - Verify all citations

2. **Report Generation**:
   - Write your final report in markdown format
   - Include all required sections:
     - Summary
     - Findings
     - Citations
     - Recommendations
   - Save to: `output/reports/markdown/researcher_$JOB_ID.md`

3. **CRITICAL FINAL STEP - Post Report to Database**:
   ```bash
   node scripts/post-report.js $JOB_ID output/reports/markdown/researcher_$JOB_ID.md
   ```
   
   ⚠️ **DO NOT SKIP THIS STEP** - Without this, your report won't be stored in the database!

4. **Verification**:
   - Confirm report was posted successfully
   - Check for success message: "✅ Report posted successfully!"

**OUTPUT REQUIREMENTS**:
- Final markdown report file
- Database updated with report_content field
- Success confirmation message
```

## 🔍 What This Script Does

The `post-report.js` script:
1. ✅ Reads your final report file
2. ✅ Extracts the clean report content
3. ✅ Updates the database `report_content` field
4. ✅ Sets the `report_format` (markdown/json/html)
5. ✅ Records the `report_path` for reference
6. ✅ Updates the `last_update` timestamp

## ❌ Common Mistakes to Avoid

### Mistake 1: Forgetting to Post Report
```bash
# ❌ WRONG - Research completed but report not posted to DB
# Agent generates report but doesn't run post-report.js
```

### Mistake 2: Wrong File Path
```bash
# ❌ WRONG - File path doesn't exist
node scripts/post-report.js $JOB_ID wrong/path/report.md

# ✅ CORRECT - Use actual file path
node scripts/post-report.js $JOB_ID output/reports/markdown/researcher_${JOB_ID}.md
```

### Mistake 3: Missing JOB_ID
```bash
# ❌ WRONG - No job ID provided
node scripts/post-report.js output/reports/markdown/report.md

# ✅ CORRECT - Include job ID first
node scripts/post-report.js abc-123-def-456 output/reports/markdown/researcher_abc-123-def-456.md
```

## 📊 Database Fields Updated

When you run `post-report.js`, these fields are updated:

```sql
report_content   -- Your full report content (markdown)
report_format    -- 'markdown', 'json', or 'html'
report_path      -- File system path to report
last_update      -- Current timestamp
```

## 🎓 Example Agent Execution

```bash
# Step 1: Agent conducts research
# ... research happens ...

# Step 2: Agent generates report
# Report saved to: output/reports/markdown/researcher_abc-123.md

# Step 3: Agent posts report to database (CRITICAL!)
node scripts/post-report.js abc-123 output/reports/markdown/researcher_abc-123.md

# Expected output:
# ✅ Report posted successfully!
#    Job ID:       abc-123
#    Report Path:  output/reports/markdown/researcher_abc-123.md
#    Format:       markdown
#    Size:         7526 characters

# Step 4: Verification (optional)
node scripts/view-job.js abc-123
# Should show report preview in output
```

## 🔧 Integration with run-researcher-local.js

The main research runner automatically handles this, but if you're running agents directly:

```javascript
// After agent completes research
const jobId = process.env.JOB_ID;
const reportPath = `output/reports/markdown/${agentName}_${jobId}.md`;

// Post report to database
spawn('node', ['scripts/post-report.js', jobId, reportPath]);
```

## 📞 Troubleshooting

### "Job not found"
- Check that JOB_ID exists in database
- Run: `node scripts/list-jobs.js` to see all jobs

### "Report file not found"
- Verify the report file was created
- Check the file path is correct
- Ensure report was saved before posting

### "Report file is empty"
- Verify your report generation completed
- Check the file contains content
- Re-generate report if needed

## ✅ Success Criteria

You've successfully posted your report when:
1. ✅ Post script shows "Report posted successfully!"
2. ✅ Database size increases (report_content populated)
3. ✅ View-job shows report preview
4. ✅ Report length matches your generated file

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│  AGENT FINAL STEP - POST REPORT TO DATABASE    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Command:                                       │
│  node scripts/post-report.js \                  │
│       $JOB_ID \                                 │
│       output/reports/markdown/${AGENT}_${JOB_ID}.md
│                                                 │
│  Variables:                                     │
│  • $JOB_ID - Your unique job identifier        │
│  • $AGENT - Your agent name (e.g., researcher) │
│                                                 │
│  Success Message:                               │
│  ✅ Report posted successfully!                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

**Remember**: The database stores TWO things separately:
- `execution_log` - Contains all stdout/stderr from the agent run
- `report_content` - Contains your FINAL CLEAN REPORT (must post manually)

**The `post-report.js` script updates the `report_content` field with your clean report!**
