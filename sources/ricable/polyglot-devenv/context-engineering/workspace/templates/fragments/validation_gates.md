## Validation Gates Fragment

### Level 1: Environment-Specific Syntax & Style
```bash
cd {ENVIRONMENT} && devbox shell
{FORMAT_COMMAND}  # Format code
{LINT_COMMAND}    # Lint code
{TYPE_CHECK_COMMAND}  # Type checking (if applicable)
```

### Level 2: Unit Tests
```bash
{TEST_COMMAND}  # Run tests with coverage
# Expected: All tests pass, adequate coverage
```

### Level 3: Integration & Cross-Environment Tests
```bash
# Single environment integration
cd {ENVIRONMENT} && devbox shell
{INTEGRATION_TEST_COMMAND}  # If available

# Cross-environment validation (if applicable)
nu nushell-env/scripts/validate-all.nu parallel

# Security and performance checks
nu nushell-env/scripts/security-scanner.nu scan-all --quiet
nu nushell-env/scripts/performance-analytics.nu measure "feature-validation"
```

## Final Validation Checklist
- [ ] Environment setup successful: `devbox shell` works
- [ ] All tests pass: `{TEST_COMMAND}`
- [ ] No linting errors: `{LINT_COMMAND}`
- [ ] No format issues: `{FORMAT_COMMAND}`
- [ ] Manual test successful: [environment-specific command]
- [ ] Cross-environment integration (if applicable)
- [ ] Security scan clean
- [ ] Performance within limits
- [ ] Error cases handled gracefully
- [ ] Documentation updated if needed