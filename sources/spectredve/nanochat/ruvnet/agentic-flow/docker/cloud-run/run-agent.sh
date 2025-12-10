#!/bin/bash
# Quick helper script to run agents on Cloud Run
# Usage: ./run-agent.sh <model> <agent> <task>
# Example: ./run-agent.sh llama coder "Create Python hello world"

set -e

MODEL="${1}"
AGENT="${2}"
TASK="${3}"
REGION="${4:-us-central1}"
PROJECT_ID="${5:-$(gcloud config get-value project)}"

if [ -z "$MODEL" ] || [ -z "$AGENT" ] || [ -z "$TASK" ]; then
  echo "Usage: $0 <model> <agent> <task> [region] [project-id]"
  echo ""
  echo "Models: claude, llama, deepseek, gemini"
  echo "Agents: coder, researcher, reviewer, planner, tester, etc."
  echo ""
  echo "Examples:"
  echo "  $0 llama coder 'Create Python hello world'"
  echo "  $0 claude researcher 'Research Python frameworks'"
  echo "  $0 deepseek reviewer 'Review code in src/index.ts'"
  exit 1
fi

JOB_NAME="agentic-flow-${MODEL}-job"

echo "🚀 Executing Agentic Flow Agent"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model:   ${MODEL}"
echo "Agent:   ${AGENT}"
echo "Task:    ${TASK}"
echo "Job:     ${JOB_NAME}"
echo "Region:  ${REGION}"
echo ""

# Execute the job
echo "⏳ Starting execution..."
EXECUTION_NAME=$(gcloud run jobs execute ${JOB_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --args="--agent,${AGENT},--task,${TASK}" \
  --format="value(metadata.name)" \
  --wait 2>&1 | tail -1)

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ Job execution completed!"
  echo ""
  echo "📋 View logs:"
  echo "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --limit=50"
  echo ""
  echo "📊 Execution details:"
  echo "  gcloud run jobs executions describe ${EXECUTION_NAME} --region=${REGION}"
else
  echo ""
  echo "❌ Job execution failed!"
  echo "View logs for details:"
  echo "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --limit=50"
  exit 1
fi
