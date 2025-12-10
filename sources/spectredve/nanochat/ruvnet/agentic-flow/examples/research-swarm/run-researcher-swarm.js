#!/usr/bin/env node

/**
 * Supabase-Integrated Researcher Agent Wrapper
 *
 * Wraps the optimized agent runner with real-time progress updates to Supabase.
 * This allows the frontend to receive live progress updates via Supabase Realtime.
 *
 * Environment Variables Required:
 * - SUPABASE_URL: Supabase project URL
 * - SUPABASE_SERVICE_ROLE_KEY: Service role key for database access
 * - JOB_ID: The permit_research_jobs table record ID
 * - All standard agent env vars (ANTHROPIC_API_KEY, etc.)
 *
 * Usage:
 *   SUPABASE_URL=xxx SUPABASE_SERVICE_ROLE_KEY=xxx JOB_ID=xxx node run-researcher-with-supabase.js agent-name "task"
 */

import { createClient } from '@supabase/supabase-js'
import { spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import dotenv from 'dotenv'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Load .env from project root (2 levels up from scripts directory)
const rootDir = path.join(__dirname, '../../')
dotenv.config({ path: path.join(rootDir, '.env') })

// Configuration
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.VITE_SUPABASE_URL
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY
let JOB_ID = process.env.JOB_ID

// Track whether JOB_ID was provided by Edge Function (vs auto-generated)
const JOB_ID_FROM_EDGE_FUNCTION = !!JOB_ID

// Get command line args
const agentName = process.argv[2]
const task = process.argv[3]

if (!agentName || !task) {
  console.error('Usage: node run-researcher-with-supabase.js <agent-name> "<task>"')
  process.exit(1)
}

// Auto-generate JOB_ID if not provided (UUID v4 format)
if (!JOB_ID && SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY) {
  // Generate a valid UUID v4
  JOB_ID = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
  console.log('🆔 Auto-generated JOB_ID (UUID):', JOB_ID)
}

// Initialize Supabase client if credentials provided
let supabase = null
let realtimeChannel = null
let hasSupabase = false

if (SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY && JOB_ID) {
  try {
    supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    hasSupabase = true
    console.log('✅ Supabase client initialized for job:', JOB_ID)

    // Initialize Realtime channel for streaming logs
    realtimeChannel = supabase.channel(`job-stream-${JOB_ID}`)
    realtimeChannel.subscribe((status) => {
      if (status === 'SUBSCRIBED') {
        console.log('✅ Realtime channel subscribed:', `job-stream-${JOB_ID}`)
      }
    })
  } catch (error) {
    console.warn('⚠️  Failed to initialize Supabase client:', error.message)
    console.warn('⚠️  Continuing without database updates')
  }
} else {
  console.log('ℹ️  No Supabase credentials provided - running without database updates')
  if (SUPABASE_URL) console.log('   SUPABASE_URL: ✅')
  if (SUPABASE_SERVICE_ROLE_KEY) console.log('   SUPABASE_SERVICE_ROLE_KEY: ✅')
  if (JOB_ID) console.log('   JOB_ID: ✅')
}

/**
 * Create initial job record in database
 */
async function createJobRecord() {
  if (!hasSupabase) {
    return
  }

  try {
    const { error: insertError } = await supabase
      .from('permit_research_jobs')
      .insert({
        id: JOB_ID,
        agent: agentName,
        task: task,
        location: task, // Using task as location for now
        status: 'running',
        progress: 0,
        current_message: 'Job initialized',
        created_at: new Date().toISOString()
      })

    if (insertError) {
      console.warn('⚠️  Failed to create job record:', insertError.message)
      console.warn('⚠️  Will continue with updates only')
    } else {
      console.log('✅ Job record created in database')
    }
  } catch (error) {
    console.warn('⚠️  Error creating job record:', error.message)
  }
}

/**
 * Update job progress in database
 * ✅ WITH BASELINE OFFSET - Prevents backwards progress jumps
 */
let baselineProgress = 0; // Track baseline to prevent backwards jumps

async function updateProgress(progress, message, additionalData = {}) {
  // Enforce baseline offset to prevent backwards jumps (10% → 30% → 15% bug)
  const safeProgress = Math.max(progress, baselineProgress);
  baselineProgress = safeProgress; // Update baseline

  console.log(`📊 Progress [${safeProgress}%]: ${message}`)

  if (!hasSupabase) {
    return
  }

  try {
    const updateData = {
      progress: Math.min(safeProgress, 95), // Max 95 until complete
      current_message: message,
      last_update: new Date().toISOString(),
      ...additionalData
    }

    const { error } = await supabase
      .from('permit_research_jobs')
      .update(updateData)
      .eq('id', JOB_ID)

    if (error) {
      console.error('❌ Failed to update progress:', error.message)
    } else {
      console.log(`✅ Database updated: ${safeProgress}%`)
    }
  } catch (error) {
    console.error('❌ Error updating progress:', error.message)
  }
}

/**
 * Clean report content by removing execution metadata and extracting the actual report
 */
function cleanReportContent(content) {
  const separator = '═══════════════════════════════════════'
  const firstIndex = content.indexOf(separator)

  let extracted = content

  if (firstIndex === -1) {
    // No separator found - use full content
    console.log('⚠️  No separator found, will search for markdown header')
  } else {
    const afterFirstSeparator = firstIndex + separator.length
    const secondIndex = content.indexOf(separator, afterFirstSeparator)

    if (secondIndex === -1) {
      // Only one separator - get everything after it
      extracted = content.substring(afterFirstSeparator).trim()
      console.log(`✅ Extracted content after separator: ${extracted.length} characters`)
    } else {
      // Two separators - get content BETWEEN them
      extracted = content.substring(afterFirstSeparator, secondIndex).trim()
      console.log(`✅ Extracted content between separators: ${extracted.length} characters`)
    }
  }

  // CRITICAL: Always remove any narrative text before the first markdown header
  const markdownStart = extracted.indexOf('# ')
  if (markdownStart > 0) {
    const beforeHeader = extracted.substring(0, markdownStart).trim()
    if (beforeHeader.length > 0) {
      console.log(`🧹 Removing ${beforeHeader.length} chars of narrative before markdown header`)
      console.log(`   Removed text: "${beforeHeader.substring(0, 80)}..."`)
      extracted = extracted.substring(markdownStart).trim()
    }
  } else if (markdownStart === -1) {
    console.log('⚠️  No markdown header (# ) found in extracted content')
  }

  console.log(`✅ Final cleaned report: ${extracted.length} characters`)
  return extracted
}

/**
 * Mark job as completed with full report and execution logs
 */
async function markComplete(exitCode, executionLog = '', reportPath = null) {
  console.log(`\n${'='.repeat(80)}`)
  console.log(`Job ${exitCode === 0 ? 'COMPLETED' : 'FAILED'} with exit code: ${exitCode}`)
  console.log('='.repeat(80))

  if (!hasSupabase) {
    return
  }

  try {
    // Read FULL report from markdown file if available
    let fullReport = null
    let reportFormat = null

    if (reportPath && fs.existsSync(reportPath)) {
      const rawReport = fs.readFileSync(reportPath, 'utf8')
      reportFormat = reportPath.endsWith('.md') ? 'markdown' : 'json'
      console.log(`✅ Full report loaded from file: ${rawReport.length} characters`)

      // Clean the report to remove execution metadata
      fullReport = cleanReportContent(rawReport)
      console.log(`✅ Report cleaned: ${fullReport.length} characters (removed ${rawReport.length - fullReport.length} chars of metadata)`)
    } else {
      console.log('⚠️  No report file found, extracting clean report from execution log')

      // Extract clean report content from execution log (between separators)
      const separator = '═══════════════════════════════════════'
      const firstIndex = executionLog.indexOf(separator)

      if (firstIndex === -1) {
        console.log('⚠️  No separator found, using full execution log')
        fullReport = executionLog
      } else {
        const afterFirstSeparator = firstIndex + separator.length
        const secondIndex = executionLog.indexOf(separator, afterFirstSeparator)

        if (secondIndex === -1) {
          // Only one separator - get everything after it
          fullReport = executionLog.substring(afterFirstSeparator).trim()
          console.log(`✅ Extracted report after separator: ${fullReport.length} characters`)
        } else {
          // Two separators - get content BETWEEN them (this is the clean report)
          fullReport = executionLog.substring(afterFirstSeparator, secondIndex).trim()
          console.log(`✅ Extracted clean report between separators: ${fullReport.length} characters`)
        }
      }

      reportFormat = 'markdown'
    }

    // Calculate duration (with timeout to prevent hanging)
    let durationSeconds = null
    try {
      const startTime = await Promise.race([
        getJobStartTime(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000))
      ])
      durationSeconds = Math.round((Date.now() - Date.parse(startTime)) / 1000) || null
    } catch (err) {
      console.warn('⚠️  Could not calculate duration:', err.message)
      durationSeconds = null
    }

    const { error } = await supabase
      .from('permit_research_jobs')
      .update({
        status: exitCode === 0 ? 'completed' : 'failed',
        progress: 100,
        current_message: exitCode === 0 ? 'Research completed successfully' : `Job failed with exit code ${exitCode}`,
        exit_code: exitCode,
        execution_log: executionLog,  // ← Execution logs (stdout)
        report_content: fullReport,   // ← Full detailed report from file
        report_format: reportFormat,
        duration_seconds: durationSeconds,
        completed_at: new Date().toISOString()
      })
      .eq('id', JOB_ID)

    if (error) {
      console.error('❌ Failed to mark job complete:', error.message)
    } else {
      console.log('✅ Job marked as complete in database')
      console.log(`   - Execution log: ${executionLog.length} chars`)
      console.log(`   - Full report: ${fullReport?.length || 0} chars`)
      console.log(`   - Duration: ${durationSeconds}s`)
    }
  } catch (error) {
    console.error('❌ Error marking job complete:', error.message)
  }
}

/**
 * Get job start time from database
 */
async function getJobStartTime() {
  if (!hasSupabase) {
    return new Date().toISOString()
  }

  try {
    const { data } = await supabase
      .from('permit_research_jobs')
      .select('started_at')
      .eq('id', JOB_ID)
      .single()

    return data?.started_at || new Date().toISOString()
  } catch (error) {
    return new Date().toISOString()
  }
}

/**
 * Main execution
 */
async function main() {
  const startTime = Date.now()

  try {
    // NOTE: Edge Function already created the job record, so we only need to update it
    // If JOB_ID was provided by Edge Function, the job already exists in the database
    // Only create a new job if JOB_ID was auto-generated (not provided by Edge Function)
    if (JOB_ID_FROM_EDGE_FUNCTION) {
      console.log('✅ Using existing job record from Edge Function:', JOB_ID)
      console.log('⏭️  Skipping createJobRecord() - job already exists')
    } else {
      console.log('🆔 Auto-generated JOB_ID - creating new job record')
      await createJobRecord()
    }

    await updateProgress(15, 'Initializing AI research agent...')

    // Path to the original agent script
    const originalScript = path.join(__dirname, 'run-researcher-agentic-flow-optimized.js')

    if (!fs.existsSync(originalScript)) {
      throw new Error(`Original agent script not found: ${originalScript}`)
    }

    await updateProgress(20, 'Starting permit research agent...')

    console.log('\n🚀 Spawning agent process...')
    console.log(`Agent: ${agentName}`)
    console.log(`Task: ${task}`)
    console.log(`Script: ${originalScript}`)
    console.log(`JOB_ID: ${JOB_ID} (will be used for report naming)\n`)

    // Spawn the original agent script with JOB_ID in environment
    const agentProcess = spawn('node', [originalScript, agentName, task], {
      env: {
        ...process.env,
        JOB_ID: JOB_ID,  // Explicitly pass JOB_ID for report naming
        WRAPPER_HANDLES_UPLOAD: 'true'  // Tell agent to skip its own Supabase upload
      },
      stdio: ['inherit', 'pipe', 'pipe']
    })

    let executionLogBuffer = ''  // ← Capture all execution output
    let lastProgressUpdate = Date.now()
    let lastToolUpdate = Date.now()

    // CRITICAL: Start at 20% because we wrote 15% then 20% above (lines 333, 342)
    // This prevents tool-based estimates from writing lower values like 5%, 10%, 15%
    let currentProgress = 20
    let lastWrittenProgress = 20  // Track what we actually wrote to DB
    let progressStage = 'initializing'

    // ⚠️  FALLBACK progress monitoring DISABLED
    // The agent script (run-researcher-agentic-flow-optimized.js) now handles ALL progress updates
    // with its own time-based monitoring system (lines 755-772).
    // This prevents duplicate/conflicting progress updates from multiple sources.
    //
    // Progress flow:
    //   Wrapper: 15% → 20% (initialization)
    //   Agent:   25% → 35% → 45% → 55% → 65% → 75% → 85% (time-based)
    //   Agent:   95% (finalization)
    //   Wrapper: 95% → 100% (completion)
    //
    // No fallback interval needed - agent provides continuous progress updates

    /**
     * Broadcast log message to Realtime channel
     */
    async function broadcastLog(message, type = 'stdout') {
      if (realtimeChannel) {
        try {
          await realtimeChannel.send({
            type: 'broadcast',
            event: 'log',
            payload: {
              job_id: JOB_ID,
              timestamp: new Date().toISOString(),
              type: type,
              message: message.trim()
            }
          })
        } catch (error) {
          // Silently ignore broadcast errors to not disrupt execution
        }
      }
    }

    // Monitor stdout for execution logs (capture everything)
    agentProcess.stdout.on('data', async (data) => {
      const output = data.toString()
      executionLogBuffer += output  // ← Capture for execution_log field
      process.stdout.write(output) // Echo to console

      // Broadcast to Realtime channel
      await broadcastLog(output, 'stdout')

      // Update stage based on keywords (for better messaging)
      if (output.includes('Searching') || output.includes('search')) {
        progressStage = 'searching'
      } else if (output.includes('Analyzing') || output.includes('analysis')) {
        progressStage = 'analyzing'
      } else if (output.includes('Generating') || output.includes('report')) {
        progressStage = 'generating'
      }
    })

    agentProcess.stderr.on('data', async (data) => {
      const output = data.toString()
      executionLogBuffer += output  // ← Also capture stderr
      process.stderr.write(output) // Echo to console

      // Broadcast to Realtime channel
      await broadcastLog(output, 'stderr')

      // Parse tool calls from agentic-flow v1.8.11+ streaming output
      // Format: [HH:MM:SS] 🔍 Tool call #N: ToolName
      const toolMatch = output.match(/\[(\d{2}:\d{2}:\d{2})\] 🔍 Tool call #(\d+): (.+)/)
      if (toolMatch && hasSupabase) {
        const [_, timestamp, toolNumber, toolName] = toolMatch
        const toolNum = parseInt(toolNumber)

        // Estimate progress based on tool calls (rough heuristic)
        // Typical agent uses 20-40 tool calls, so each call = ~2% progress
        // Start from 20% baseline (what we hardcoded above) to prevent backwards jumps
        const estimatedProgress = Math.min(20 + Math.floor(toolNum * 2), 90)

        // CRITICAL: Only update if progress is GREATER than last written progress
        if (estimatedProgress > lastWrittenProgress) {
          // Track tool update time to prevent time-based updates from overwriting
          lastToolUpdate = Date.now()

          // Update progress with streaming tool info
          await updateProgress(
            estimatedProgress,
            `Executing ${toolName.trim()}`,
            {
              current_tool: toolName.trim(),
              tool_count: toolNum,
              tool_timestamp: timestamp
            }
          ).catch(err => {
            // Don't fail execution if progress update fails
            console.error(`⚠️  Progress update failed: ${err.message}`)
          })

          // Update both current and last written progress
          currentProgress = estimatedProgress
          lastWrittenProgress = estimatedProgress
          lastProgressUpdate = Date.now()
        }
      }
    })

    // Wait for agent to complete
    const exitCode = await new Promise((resolve) => {
      agentProcess.on('close', resolve)
    })

    // ⚠️  No progress interval to clear - monitoring is now handled entirely by agent script

    const duration = Math.round((Date.now() - startTime) / 1000)
    console.log(`\n⏱️  Duration: ${duration}s (${(duration / 60).toFixed(2)} min)`)
    console.log(`📊 Execution log captured: ${executionLogBuffer.length} characters`)

    // Find the generated FULL report file
    const OUTPUT_DIR = path.join(__dirname, '../../output')
    let reportPath = null

    console.log(`🔍 Searching for report files with JOB_ID: ${JOB_ID}`)

    // Strategy 1: Check JSON files (priority - most reliable)
    // JSON files contain the full "result" field with complete agent output
    if (!reportPath) {
      const jsonDir = path.join(OUTPUT_DIR, 'reports/json')
      console.log(`   Strategy 1: Checking reports/json (HIGHEST PRIORITY) for JOB_ID: ${jsonDir}`)

      if (fs.existsSync(jsonDir)) {
        const jsonFiles = fs.readdirSync(jsonDir)
          .filter(f => {
            if (!f.endsWith('.json')) return false
            if (!f.includes(JOB_ID)) return false
            const fullPath = path.join(jsonDir, f)
            const stats = fs.statSync(fullPath)
            return stats.isFile() && stats.mtime.getTime() >= startTime
          })
          .map(f => ({
            path: path.join(jsonDir, f),
            time: fs.statSync(path.join(jsonDir, f)).mtime,
            size: fs.statSync(path.join(jsonDir, f)).size,
            name: f
          }))
          .sort((a, b) => b.time - a.time)

        if (jsonFiles.length > 0) {
          const jsonFile = jsonFiles[0].path
          console.log(`✅ JSON output file found: ${jsonFiles[0].name}`)

          try {
            const jsonContent = JSON.parse(fs.readFileSync(jsonFile, 'utf8'))
            if (jsonContent.result) {
              // Extract content from JSON result field and write to temp file
              const extractedContent = jsonContent.result
              console.log(`✅ Extracted ${extractedContent.length} chars from JSON result field`)

              // Write to temp markdown file for processing
              const tempMdPath = path.join(OUTPUT_DIR, 'reports/markdown', `temp_${JOB_ID}.md`)
              fs.writeFileSync(tempMdPath, extractedContent, 'utf8')
              console.log(`✅ Wrote extracted content to temp file for cleaning`)

              reportPath = tempMdPath
            }
          } catch (err) {
            console.warn(`⚠️  Failed to parse JSON file: ${err.message}`)
          }
        }
      }
    }

    // Strategy 4: Look for DESCRIPTIVE files WITHOUT JOB_ID (clean reports)
    // Files WITH JOB_ID are execution summaries with metadata - we EXCLUDE them
    if (fs.existsSync(OUTPUT_DIR)) {
      console.log(`   Checking root for descriptive files (NOT JOB_ID): ${OUTPUT_DIR}`)
      const descriptiveFiles = fs.readdirSync(OUTPUT_DIR)
        .filter(f => {
          if (!f.endsWith('.md') || f.includes('SUMMARY')) return false
          // EXCLUDE files with JOB_ID (those are execution summaries)
          if (f.includes(JOB_ID)) {
            console.log(`   ⏭️  Skipping JOB_ID file (execution summary): ${f}`)
            return false
          }
          // Must contain part of agent name
          const agentPart = agentName.replace('-researcher', '').replace(/-/g, '_')
          if (!f.includes(agentPart)) return false

          const fullPath = path.join(OUTPUT_DIR, f)
          const stats = fs.statSync(fullPath)
          // Only files created after job start
          if (!stats.isFile() || stats.mtime.getTime() < startTime) return false
          return true
        })
        .map(f => ({
          path: path.join(OUTPUT_DIR, f),
          time: fs.statSync(path.join(OUTPUT_DIR, f)).mtime,
          size: fs.statSync(path.join(OUTPUT_DIR, f)).size,
          name: f
        }))
        .sort((a, b) => b.time - a.time)

      if (descriptiveFiles.length > 0) {
        reportPath = descriptiveFiles[0].path
        console.log(`✅ Full report file found (descriptive, NOT JOB_ID): ${descriptiveFiles[0].name} (${descriptiveFiles[0].size} bytes)`)
      }
    }

    // Strategy 2: Look in reports/markdown WITHOUT JOB_ID (clean reports)
    if (!reportPath) {
      const reportsDir = path.join(OUTPUT_DIR, 'reports/markdown')
      console.log(`   Checking reports/markdown for descriptive files (NOT JOB_ID): ${reportsDir}`)

      if (fs.existsSync(reportsDir)) {
        const files = fs.readdirSync(reportsDir)
          .filter(f => {
            if (!f.endsWith('.md') || f.includes('SUMMARY')) return false
            // EXCLUDE files with JOB_ID
            if (f.includes(JOB_ID)) {
              console.log(`   ⏭️  Skipping JOB_ID file (execution summary): ${f}`)
              return false
            }
            // Must contain part of agent name
            const agentPart = agentName.replace('-researcher', '').replace(/-/g, '_')
            if (!f.includes(agentPart)) return false

            const fullPath = path.join(reportsDir, f)
            const stats = fs.statSync(fullPath)
            // Only files created after job start
            if (!stats.isFile() || stats.mtime.getTime() < startTime) return false
            return true
          })
          .map(f => ({
            path: path.join(reportsDir, f),
            time: fs.statSync(path.join(reportsDir, f)).mtime,
            size: fs.statSync(path.join(reportsDir, f)).size,
            name: f
          }))
          .sort((a, b) => b.time - a.time)

        if (files.length > 0) {
          reportPath = files[0].path
          console.log(`✅ Full report file found (descriptive, NOT JOB_ID): ${files[0].name} (${files[0].size} bytes)`)
        }
      }
    }

    // Strategy 3: Look for agent-specific files created after job start (fallback)
    if (!reportPath && fs.existsSync(OUTPUT_DIR)) {
      console.log(`   Fallback: Looking for ${agentName} files created after ${new Date(startTime).toISOString()}`)
      const agentFiles = fs.readdirSync(OUTPUT_DIR)
        .filter(f => {
          if (!f.endsWith('.md') || f.includes('SUMMARY')) return false
          if (!f.includes(agentName)) return false
          const fullPath = path.join(OUTPUT_DIR, f)
          const stats = fs.statSync(fullPath)
          // Only files created after this job started
          return stats.isFile() && stats.mtime.getTime() >= startTime
        })
        .map(f => ({
          path: path.join(OUTPUT_DIR, f),
          time: fs.statSync(path.join(OUTPUT_DIR, f)).mtime,
          size: fs.statSync(path.join(OUTPUT_DIR, f)).size,
          name: f
        }))
        .sort((a, b) => b.time - a.time)

      if (agentFiles.length > 0) {
        reportPath = agentFiles[0].path
        console.log(`✅ Full report file found by agent name: ${agentFiles[0].name} (${agentFiles[0].size} bytes)`)
      }
    }

    // Strategy 5: Look for JOB_ID markdown files (less reliable fallback)
    // Note: These may contain prompt instructions instead of actual output
    if (!reportPath) {
      const reportsDir = path.join(OUTPUT_DIR, 'reports/markdown')
      console.log(`   Fallback: Checking reports/markdown for JOB_ID files: ${reportsDir}`)

      if (fs.existsSync(reportsDir)) {
        const jobIdFiles = fs.readdirSync(reportsDir)
          .filter(f => {
            if (!f.endsWith('.md')) return false
            if (!f.includes(JOB_ID)) return false
            if (f.startsWith('temp_')) return false // Skip temp files we created
            const fullPath = path.join(reportsDir, f)
            const stats = fs.statSync(fullPath)
            return stats.isFile() && stats.mtime.getTime() >= startTime
          })
          .map(f => ({
            path: path.join(reportsDir, f),
            time: fs.statSync(path.join(reportsDir, f)).mtime,
            size: fs.statSync(path.join(reportsDir, f)).size,
            name: f
          }))
          .sort((a, b) => b.time - a.time)

        if (jobIdFiles.length > 0) {
          reportPath = jobIdFiles[0].path
          console.log(`⚠️  Using markdown JOB_ID file (may need extra cleaning): ${jobIdFiles[0].name} (${jobIdFiles[0].size} bytes)`)
        }
      }
    }

    // Fallback: Use execution output if NO report file found
    if (!reportPath) {
      console.warn('⚠️  No full report markdown file found anywhere')
      console.warn('⚠️  Will extract report content from execution log as fallback')
      console.warn('⚠️  Note: This may contain only the summary, not the full detailed report')

      if (exitCode === 0) {
        console.warn('⚠️  IMPORTANT: Agent exited successfully (code 0) but no report file found!')
        console.warn('⚠️  This suggests the agent may not have completed its file writing.')
        console.warn('⚠️  Consider increasing job timeout or debugging file generation.')
      }
    }

    // Update progress to 95% before marking complete
    await updateProgress(95, 'Finalizing report and updating database...').catch(err => {
      console.warn('⚠️  Failed to update progress to 95%:', err.message)
    })

    // Broadcast completion status
    await broadcastLog('Agent process completed, finalizing database update...', 'info').catch(() => {})

    // Mark job as complete with BOTH execution log and full report
    await markComplete(exitCode, executionLogBuffer, reportPath)

    process.exit(exitCode)

  } catch (error) {
    console.error('\n❌ Agent execution failed:', error.message)
    console.error(error.stack)
    const errorLog = `Error: ${error.message}\n${error.stack}`
    await updateProgress(0, `Error: ${error.message}`, {
      status: 'failed',
      exit_code: 1
    })
    await markComplete(1, errorLog, null)
    process.exit(1)
  }
}

// Run main function
main().catch(error => {
  console.error('Fatal error:', error)
  process.exit(1)
})
