#!/usr/bin/env node
import { getJobs } from '../lib/db-utils.js';

const jobs = getJobs({ limit: 20 });

console.log(`\n📋 Recent Jobs (${jobs.length}):\n`);

if (jobs.length === 0) {
  console.log('   No jobs found.\n');
} else {
  jobs.forEach(job => {
    const status = job.status === 'completed' ? '✅' :
                   job.status === 'running' ? '🔄' :
                   job.status === 'failed' ? '❌' : '⏸️';
    console.log(`${status} ${job.id.substring(0, 8)} | ${job.agent} | ${job.progress}% | ${job.status}`);
    console.log(`   Task: ${job.task.substring(0, 60)}...`);
    console.log(`   Created: ${job.created_at}\n`);
  });
}
