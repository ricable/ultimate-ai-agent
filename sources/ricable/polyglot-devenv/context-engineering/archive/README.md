# Context Engineering Archive

The **archive** directory contains historical PRPs, execution reports, and analysis data for tracking the evolution of the context engineering system.

## Structure

- **PRPs/** - Previously generated PRPs that have been executed
  - `multi_agent_prp.md` - Multi-agent system implementation
  - `user-api-multi.md` - Cross-environment user API implementation  
  - `user-api-python.md` - Python-specific user API implementation
  - `templates/prp_base.md` - Legacy base template
- **execution-reports/** - Historical execution reports and performance data
- **analysis/** - Performance analysis and trend data
- **versions/** - Versioned snapshots of major system changes

## Purpose

The archive serves multiple purposes:

1. **Historical Tracking**: Maintain record of all PRPs and their evolution
2. **Performance Analysis**: Track system performance over time
3. **Pattern Recognition**: Identify successful patterns and common issues
4. **Regression Testing**: Reference implementations for testing changes
5. **Learning Resource**: Examples of working PRPs for training

## Archival Process

### Automatic Archiving

PRPs are automatically archived when:
1. **Execution Completes**: Successfully executed PRPs move from workspace to archive
2. **System Reorganization**: During major restructuring (like this reorganization)
3. **Version Updates**: When PRP templates or system architecture changes

### Manual Archiving

```bash
# Archive a completed PRP
mv context-engineering/workspace/PRPs/feature-python.md context-engineering/archive/PRPs/

# Archive execution reports
mv context-engineering/devpod/execution/reports/* context-engineering/archive/execution-reports/

# Create version snapshot
cp -r context-engineering/workspace/templates context-engineering/archive/versions/templates-$(date +%Y%m%d)
```

## Historical Analysis

### Performance Tracking

```bash
# Analyze historical performance
nu dev-env/nushell/scripts/performance-analytics.nu report --source archive --days 30

# Compare PRP evolution
nu shared/utils/template-engine.nu --compare archive/PRPs/user-api-python.md workspace/PRPs/user-api-v2-python.md
```

### Pattern Analysis

The archive enables analysis of:
- **Success Patterns**: Which PRP structures work best
- **Common Failures**: Recurring issues and their solutions
- **Evolution Trends**: How PRPs improve over time
- **Environment Differences**: Success rates by environment

## Access Patterns

### Research and Reference

```bash
# Search archived PRPs for patterns
grep -r "FastAPI" context-engineering/archive/PRPs/

# Find similar implementations
nu shared/utils/argument-parser.nu --search archive --pattern "user management"

# Compare template versions
diff context-engineering/archive/versions/templates-20250107/python_prp.md context-engineering/workspace/templates/python_prp.md
```

### Personal Learning

```bash
# Personal aliases for archive access (add to CLAUDE.local.md)
alias prp-history="ls context-engineering/archive/PRPs/"
alias prp-compare="diff context-engineering/archive/PRPs/\$1 context-engineering/workspace/PRPs/\$1"
alias prp-learn="grep -r \$1 context-engineering/archive/PRPs/"
```

## Maintenance

### Retention Policy

- **Active PRPs**: Keep indefinitely for reference
- **Execution Reports**: Retain for 1 year, then compress
- **Performance Data**: Keep aggregated data, archive raw data after 6 months
- **Version Snapshots**: Keep major versions, monthly snapshots for 1 year

### Cleanup Commands

```bash
# Compress old execution reports (older than 6 months)
find context-engineering/archive/execution-reports -type f -mtime +180 -exec gzip {} \;

# Archive version snapshots (older than 1 year)  
find context-engineering/archive/versions -type d -mtime +365 -exec tar -czf {}.tar.gz {} \; -exec rm -rf {} \;
```

## Integration with System

The archive integrates with:
- **Performance Analytics**: Historical data for trend analysis
- **Template Evolution**: Tracking template improvements
- **Quality Assessment**: Comparing current vs historical success rates
- **Personal Learning**: Reference implementations for skill development

## Future Enhancements

Planned archive features:
- **Automated Tagging**: Tag PRPs by success rate, complexity, environment
- **Search Interface**: Advanced search across archived content
- **Trend Analysis**: Automated reporting on patterns and improvements
- **Integration Testing**: Use archived PRPs for regression testing