#!/usr/bin/env node

/**
 * Post Final Report to Database
 * 
 * Simple script for agents to post their final report content to the database.
 * This ensures the report_content field contains the actual report, not logs.
 * 
 * Usage:
 *   node scripts/post-report.js <job-id> <report-file-path>
 *   
 * Example:
 *   node scripts/post-report.js abc-123 output/reports/markdown/researcher_abc-123.md
 */

import fs from 'fs';
import { getDatabase } from '../lib/db-utils.js';

// Get arguments
const jobId = process.argv[2];
const reportPath = process.argv[3];

if (!jobId || !reportPath) {
  console.error(`
❌ Missing required arguments

Usage:
  node scripts/post-report.js <job-id> <report-file-path>

Example:
  node scripts/post-report.js abc-123 output/reports/markdown/researcher_abc-123.md

This script updates the database with your final report content.
`);
  process.exit(1);
}

// Check if report file exists
if (!fs.existsSync(reportPath)) {
  console.error(`❌ Report file not found: ${reportPath}`);
  process.exit(1);
}

try {
  // Read report content
  const reportContent = fs.readFileSync(reportPath, 'utf8');
  
  if (!reportContent || reportContent.length === 0) {
    console.error('❌ Report file is empty');
    process.exit(1);
  }

  // Determine format from extension
  let reportFormat = 'markdown';
  if (reportPath.endsWith('.json')) reportFormat = 'json';
  else if (reportPath.endsWith('.html')) reportFormat = 'html';

  // Update database
  const db = getDatabase();
  
  const stmt = db.prepare(`
    UPDATE research_jobs
    SET report_content = ?,
        report_format = ?,
        report_path = ?,
        last_update = datetime('now')
    WHERE id = ?
  `);

  const result = stmt.run(reportContent, reportFormat, reportPath, jobId);
  
  db.close();

  if (result.changes === 0) {
    console.error(`❌ Job not found: ${jobId}`);
    process.exit(1);
  }

  console.log(`
✅ Report posted successfully!

   Job ID:       ${jobId}
   Report Path:  ${reportPath}
   Format:       ${reportFormat}
   Size:         ${reportContent.length} characters
   Updated:      ${new Date().toISOString()}

   Database:     ./data/research-jobs.db
`);

  process.exit(0);

} catch (error) {
  console.error(`❌ Error posting report: ${error.message}`);
  process.exit(1);
}
