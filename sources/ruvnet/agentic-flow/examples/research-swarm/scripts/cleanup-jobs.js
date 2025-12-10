#!/usr/bin/env node

/**
 * Cleanup Old Jobs Script
 *
 * Remove old completed/failed jobs from the database to free up space.
 *
 * Usage:
 *   node scripts/cleanup-jobs.js [--days <n>] [--dry-run]
 *   npm run cleanup -- --days 30 --dry-run
 */

const { cleanupOldJobs, getJobs } = require('../lib/db-utils');

// Parse command line arguments
const args = process.argv.slice(2);
let olderThanDays = 30;
let dryRun = false;

for (let i = 0; i < args.length; i++) {
    if (args[i] === '--days' && args[i + 1]) {
        olderThanDays = parseInt(args[i + 1], 10);
        i++;
    } else if (args[i] === '--dry-run') {
        dryRun = true;
    }
}

console.log('🧹 Database Cleanup Utility');
console.log('');
console.log(`Configuration:`);
console.log(`  Delete jobs older than: ${olderThanDays} days`);
console.log(`  Statuses to delete: completed, failed`);
console.log(`  Mode: ${dryRun ? 'DRY RUN (no changes)' : 'LIVE (will delete)'}`);
console.log('');

try {
    // Get count of jobs that would be deleted
    const allJobs = getJobs({ limit: 10000 });
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - olderThanDays);

    const jobsToDelete = allJobs.filter(job => {
        const jobDate = new Date(job.created_at);
        return (job.status === 'completed' || job.status === 'failed') &&
               jobDate < cutoffDate;
    });

    if (jobsToDelete.length === 0) {
        console.log('✅ No jobs found matching cleanup criteria.');
        console.log('');
        console.log('Current job distribution:');
        const statusCounts = {};
        allJobs.forEach(job => {
            statusCounts[job.status] = (statusCounts[job.status] || 0) + 1;
        });
        Object.entries(statusCounts).forEach(([status, count]) => {
            console.log(`  ${status}: ${count}`);
        });
        process.exit(0);
    }

    console.log(`Found ${jobsToDelete.length} job(s) to delete:`);
    console.log('');

    // Show breakdown by status
    const statusBreakdown = {};
    jobsToDelete.forEach(job => {
        statusBreakdown[job.status] = (statusBreakdown[job.status] || 0) + 1;
    });

    Object.entries(statusBreakdown).forEach(([status, count]) => {
        console.log(`  ${status}: ${count} job(s)`);
    });
    console.log('');

    // Show sample of jobs
    console.log('Sample jobs to be deleted:');
    jobsToDelete.slice(0, 5).forEach(job => {
        console.log(`  - ${job.id.substring(0, 8)} | ${job.agent} | ${job.status} | ${new Date(job.created_at).toLocaleDateString()}`);
    });
    if (jobsToDelete.length > 5) {
        console.log(`  ... and ${jobsToDelete.length - 5} more`);
    }
    console.log('');

    if (dryRun) {
        console.log('🔍 DRY RUN MODE - No changes made to database');
        console.log('');
        console.log('To actually delete these jobs, run:');
        console.log(`  npm run cleanup -- --days ${olderThanDays}`);
    } else {
        // Perform actual cleanup
        const deletedCount = cleanupOldJobs({ olderThanDays });
        console.log(`✅ Successfully deleted ${deletedCount} job(s)`);
        console.log('');

        // Show remaining jobs
        const remainingJobs = getJobs({ limit: 10000 });
        console.log(`Remaining jobs in database: ${remainingJobs.length}`);
    }

} catch (error) {
    console.error('❌ Cleanup failed:', error.message);
    process.exit(1);
}
