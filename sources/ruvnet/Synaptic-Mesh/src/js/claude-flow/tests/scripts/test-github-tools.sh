#!/bin/bash
source test-mcp-tools-simple.sh
CATEGORY="github"

log_info "🐙 Testing GitHub Integration Tools (8 tools)"
test_tool "github_repo_analyze" '{"repo":"test/repo","analysis_type":"code_quality"}'
test_tool "github_pr_manage" '{"repo":"test/repo","action":"review","pr_number":1}'
test_tool "github_issue_track" '{"repo":"test/repo","action":"list"}'
test_tool "github_release_coord" '{"repo":"test/repo","version":"v1.0.0"}'
test_tool "github_workflow_auto" '{"repo":"test/repo","workflow":{"name":"ci"}}'
test_tool "github_code_review" '{"repo":"test/repo","pr":1}'
test_tool "github_sync_coord" '{"repos":["repo1","repo2"]}'
test_tool "github_metrics" '{"repo":"test/repo"}'
log_success "🐙 GitHub category testing complete!"