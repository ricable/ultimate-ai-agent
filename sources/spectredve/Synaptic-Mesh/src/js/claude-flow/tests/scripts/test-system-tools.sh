#!/bin/bash
source test-mcp-tools-simple.sh
CATEGORY="system"

log_info "⚙️ Testing System & Utilities Tools (8 tools)"
test_tool "terminal_execute" '{"command":"echo","args":["hello"]}'
test_tool "config_manage" '{"action":"get","config":{"section":"general"}}'
test_tool "features_detect" '{"component":"mcp_server"}'
test_tool "security_scan" '{"target":"localhost","depth":"basic"}'
test_tool "backup_create" '{"components":["config","data"],"destination":"/tmp/backup"}'
test_tool "restore_system" '{"backupId":"backup_123"}'
test_tool "log_analysis" '{"logFile":"/var/log/app.log","patterns":["ERROR","WARN"]}'
test_tool "diagnostic_run" '{"components":["network","disk","memory"]}'
log_success "⚙️ System category testing complete!"