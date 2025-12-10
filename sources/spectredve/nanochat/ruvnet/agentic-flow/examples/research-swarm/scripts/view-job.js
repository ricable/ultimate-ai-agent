#!/usr/bin/env node
import { getJobStatus } from '../lib/db-utils.js';

const jobId = process.argv[2];

if (!jobId) {
  console.error('Usage: node scripts/view-job.js <job-id>');
  process.exit(1);
}

const job = getJobStatus(jobId);

if (!job) {
  console.error(`Job not found: ${jobId}`);
  process.exit(1);
}

console.log(`
╔═══════════════════════════════════════════════════════════════════╗
║                           JOB DETAILS                             ║
╚═══════════════════════════════════════════════════════════════════╝

🆔 ID:          ${job.id}
🤖 Agent:       ${job.agent}
📋 Task:        ${job.task}
📊 Status:      ${job.status}
📈 Progress:    ${job.progress}%
💬 Message:     ${job.current_message || 'N/A'}

⏱️  TIMING:
   Created:     ${job.created_at}
   Started:     ${job.started_at || 'N/A'}
   Completed:   ${job.completed_at || 'N/A'}
   Duration:    ${job.duration_seconds ? `${job.duration_seconds}s` : 'N/A'}

📄 REPORT:
   Format:      ${job.report_format || 'N/A'}
   Path:        ${job.report_path || 'N/A'}
   Size:        ${job.report_content ? `${job.report_content.length} chars` : 'N/A'}

${job.exit_code !== null ? `Exit Code:   ${job.exit_code}` : ''}
${job.error_message ? `Error:       ${job.error_message}` : ''}

${'═'.repeat(71)}
`);

if (job.report_content && job.report_content.length > 0) {
  console.log('📄 REPORT PREVIEW (first 500 chars):\n');
  console.log(job.report_content.substring(0, 500) + '...\n');
}
