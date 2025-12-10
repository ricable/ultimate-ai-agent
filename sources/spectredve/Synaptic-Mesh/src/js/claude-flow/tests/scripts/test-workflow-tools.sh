#!/bin/bash
source test-mcp-tools-simple.sh
CATEGORY="workflow"

log_info "🔧 Testing Workflow & Automation Tools (11 tools)"
test_tool "workflow_create" '{"name":"test_workflow","steps":["step1","step2"],"triggers":["manual"]}'
test_tool "workflow_execute" '{"workflowId":"test_workflow","params":{}}'
test_tool "workflow_export" '{"workflowId":"test_workflow","format":"json"}'
test_tool "sparc_mode" '{"mode":"dev","task_description":"test development task"}'
test_tool "automation_setup" '{"rules":[{"trigger":"event","action":"response"}]}'
test_tool "pipeline_create" '{"config":{"stages":["build","test","deploy"]}}'
test_tool "scheduler_manage" '{"action":"create","schedule":{"cron":"0 0 * * *"}}'
test_tool "trigger_setup" '{"events":["push"],"actions":["build"]}'
test_tool "workflow_template" '{"action":"create","template":{"name":"ci_template"}}'
test_tool "batch_process" '{"items":["item1","item2"],"operation":"transform"}'
test_tool "parallel_execute" '{"tasks":[{"id":"task1"},{"id":"task2"}]}'
log_success "🔧 Workflow category testing complete!"