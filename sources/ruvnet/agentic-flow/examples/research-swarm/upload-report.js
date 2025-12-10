#!/usr/bin/env node

/**
 * Simple helper script for agents to upload reports directly to Supabase
 *
 * Usage: node upload-report-to-supabase.js <report-file-path>
 *
 * Requires environment variables:
 * - SUPABASE_URL
 * - SUPABASE_SERVICE_ROLE_KEY
 * - JOB_ID
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Get report file path from command line
const reportFilePath = process.argv[2];

if (!reportFilePath) {
  console.error('❌ ERROR: Report file path required');
  console.error('Usage: node upload-report-to-supabase.js <report-file-path>');
  process.exit(1);
}

// Check environment variables
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const JOB_ID = process.env.JOB_ID;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !JOB_ID) {
  console.error('❌ ERROR: Missing required environment variables');
  console.error('Required: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, JOB_ID');
  process.exit(1);
}

// Check if file exists
const fullPath = path.isAbsolute(reportFilePath)
  ? reportFilePath
  : path.join(process.cwd(), reportFilePath);

if (!fs.existsSync(fullPath)) {
  console.error(`❌ ERROR: Report file not found: ${fullPath}`);
  process.exit(1);
}

// Read report content
let reportContent;
try {
  reportContent = fs.readFileSync(fullPath, 'utf8');
} catch (error) {
  console.error(`❌ ERROR: Failed to read report file: ${error.message}`);
  process.exit(1);
}

// Determine format from extension
const reportFormat = fullPath.endsWith('.md') ? 'markdown' :
                     fullPath.endsWith('.json') ? 'json' : 'text';

console.log(`\n📤 Uploading report to Supabase...`);
console.log(`   File: ${path.basename(fullPath)}`);
console.log(`   Size: ${reportContent.length} characters`);
console.log(`   Format: ${reportFormat}`);
console.log(`   Job ID: ${JOB_ID}`);

// Create Supabase client and upload
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

supabase
  .from('permit_research_jobs')
  .update({
    report_content: reportContent,
    report_format: reportFormat,
    status: 'completed',
    progress: 100,
    current_message: 'Report uploaded successfully',
    completed_at: new Date().toISOString()
  })
  .eq('id', JOB_ID)
  .then(({ error }) => {
    if (error) {
      console.error(`\n❌ Supabase update failed: ${error.message}`);
      process.exit(1);
    } else {
      console.log(`\n✅ SUCCESS: Report uploaded to Supabase database!`);
      console.log(`   Job ID: ${JOB_ID}`);
      console.log(`   Report length: ${reportContent.length} characters`);
      console.log(`   Status: completed`);
      console.log(`   Progress: 100%\n`);
      process.exit(0);
    }
  })
  .catch((err) => {
    console.error(`\n❌ Upload error: ${err.message}`);
    process.exit(1);
  });
