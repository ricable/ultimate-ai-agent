# Claude Flow v2.0.0-alpha.52 Release Notes

## 🎉 Critical Fix for Claude Code 1.0.51+ Compatibility

This release addresses the breaking changes introduced in Claude Code 1.0.51 that affected hooks configuration and settings validation.

## ✨ What's New

### 🔧 Hooks Format Migration

**New `migrate-hooks` Command**
- Automatically converts old hook format to new Claude Code 1.0.51+ format
- Creates backups before making changes
- Removes unsupported fields (mcpServers, features, performance)
- Searches common locations automatically

```bash
# Update to latest alpha
npm install -g claude-flow@alpha

# Run the migration
claude-flow migrate-hooks

# Or with npx
npx claude-flow@alpha migrate-hooks
```

### 🛠️ Fixed Settings Generation

- Updated `init` command to generate correct hooks format
- New installations now create Claude Code 1.0.51+ compatible settings
- Proper array-based hooks structure with matchers

### 📝 Hooks Format Changes

**Old Format (No longer supported):**
```json
"hooks": {
  "preEditHook": { "command": "npx", "args": [...] },
  "postEditHook": { "command": "npx", "args": [...] }
}
```

**New Format (Required for Claude Code 1.0.51+):**
```json
"hooks": {
  "PreToolUse": [{
    "matcher": "Edit|Write",
    "hooks": [{
      "type": "command",
      "command": "npx claude-flow@alpha hooks pre-edit..."
    }]
  }]
}
```

## 🐛 Bug Fixes

- Fixed hooks format incompatibility with Claude Code 1.0.51+ (#235, #236, #237)
- Updated templates to remove unrecognized fields
- Added @alpha tag to hook commands for version consistency

## 📚 Documentation

- Added migration instructions for existing users
- Updated documentation to reflect new hooks format
- Added troubleshooting guide for settings validation errors

## 🙏 Acknowledgments

Special thanks to:
- @bailejl for reporting the issue (#235, #236)
- @jimweller for the detailed analysis and PR #238
- @guilleojeda, @pnocera, @pksorensen, and @fmal for additional insights

## 💡 Upgrade Instructions

1. **For Existing Users:**
   ```bash
   npm install -g claude-flow@alpha
   claude-flow migrate-hooks
   claude mcp add claude-flow npx claude-flow@alpha mcp start
   ```

2. **For New Users:**
   ```bash
   npm install -g claude-flow@alpha
   claude-flow init --sparc
   ```

3. **Verify Settings:**
   - Run `/doctor` in Claude Code to ensure settings are valid
   - Restart Claude Code after migration

## ⚠️ Breaking Changes

- Removed support for object-based hooks format
- MCP servers must now be added via `claude mcp add` command
- Removed `mcpServers`, `features`, and `performance` fields from settings.json

## 🔜 Coming Soon

- `enabledMcpjsonServers` support for trusted MCP servers
- Enhanced version detection for future Claude Code updates
- Improved error messages for configuration issues

---

For questions or issues, please visit: https://github.com/ruvnet/claude-flow/issues