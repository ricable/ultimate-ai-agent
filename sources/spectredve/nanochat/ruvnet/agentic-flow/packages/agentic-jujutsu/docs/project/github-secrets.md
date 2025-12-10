# GitHub Secrets Setup Guide

## Required Secrets for CI/CD Pipeline

To enable the complete CI/CD pipeline for publishing `agentic-jujutsu` to NPM, you need to configure the following GitHub secrets.

## NPM_TOKEN

### Purpose
Authenticates GitHub Actions to publish packages to NPM registry.

### Setup Steps

1. **Generate NPM Access Token**
   ```bash
   # Login to NPM
   npm login

   # Visit NPM website
   # Go to: https://www.npmjs.com/settings/YOUR_USERNAME/tokens
   ```

2. **Create Automation Token**
   - Click "Generate New Token"
   - Select "Automation" (for CI/CD use)
   - Give it a descriptive name: `GitHub Actions - agentic-jujutsu`
   - Click "Generate Token"
   - **IMPORTANT**: Copy the token immediately (shown only once)

3. **Add to GitHub Repository**
   ```bash
   # Using GitHub CLI
   gh secret set NPM_TOKEN --body "npm_xxxxxxxxxxxxxxxxxxxx"

   # Or manually:
   # 1. Go to: https://github.com/ruvnet/agentic-flow/settings/secrets/actions
   # 2. Click "New repository secret"
   # 3. Name: NPM_TOKEN
   # 4. Value: [paste your token]
   # 5. Click "Add secret"
   ```

### Token Permissions

The NPM token needs:
- ✅ **Publish**: Publish new versions
- ✅ **Read**: Read package information
- ✅ **Write**: Update package metadata

### Security Best Practices

1. **Use Automation Tokens**
   - Not your personal login token
   - Can be revoked without affecting your account
   - Designed for CI/CD use

2. **Minimum Permissions**
   - Grant only necessary permissions
   - Scope to specific packages if possible

3. **Token Rotation**
   - Rotate tokens every 90 days
   - Update GitHub secret when rotated

4. **Monitor Usage**
   - Check NPM audit logs regularly
   - Review published versions

## GITHUB_TOKEN

### Purpose
Automatically provided by GitHub Actions for repository operations.

### Configuration
**No setup required** - GitHub automatically provides this token to workflows.

### Permissions Used
- ✅ Create releases
- ✅ Upload release artifacts
- ✅ Read repository contents
- ✅ Write to repository (for tagging)

### Customizing Permissions

If needed, customize in workflow file:
```yaml
permissions:
  contents: write
  packages: write
```

## Verification

### Test Token Setup

```bash
# Clone the repository
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow/packages/agentic-jujutsu

# Create a test tag (don't push yet)
git tag v1.0.0-test

# Check if secret is configured
gh secret list

# Should show:
# NPM_TOKEN    Updated YYYY-MM-DD
```

### Test Workflow

1. **Create a test branch**
   ```bash
   git checkout -b test-ci-cd
   ```

2. **Trigger build workflow**
   ```bash
   git push origin test-ci-cd
   ```

3. **Check workflow status**
   ```bash
   gh workflow view "Build N-API Native Addons"
   gh run list --workflow=build-native.yml
   ```

4. **Test publish (dry run)**
   ```bash
   # Update version to test
   npm version prerelease --preid=test

   # Create tag
   git tag v1.0.0-test.1

   # Push tag (triggers publish workflow)
   git push origin v1.0.0-test.1

   # Monitor workflow
   gh run watch
   ```

## Troubleshooting

### NPM_TOKEN Not Working

**Error**: `npm ERR! 401 Unauthorized`

**Solutions**:
1. Verify token is correct
   ```bash
   gh secret list
   ```

2. Regenerate NPM token
   - Go to NPM settings
   - Revoke old token
   - Create new automation token
   - Update GitHub secret

3. Check token permissions
   - Ensure "Automation" type
   - Verify publish permissions

### GITHUB_TOKEN Permission Denied

**Error**: `403 Resource not accessible by integration`

**Solutions**:
1. Add permissions to workflow:
   ```yaml
   permissions:
     contents: write
     packages: write
   ```

2. Check repository settings:
   - Go to Settings > Actions > General
   - Under "Workflow permissions"
   - Select "Read and write permissions"

### Token Expiration

**Error**: `npm ERR! 401 Unauthorized` (after working previously)

**Solutions**:
1. Check token expiration
2. Regenerate NPM token
3. Update GitHub secret
4. Re-run failed workflow

## Secret Management Best Practices

### 1. Environment Separation

For different environments:
```yaml
# Production
environment: production
secrets:
  NPM_TOKEN: ${{ secrets.NPM_TOKEN }}

# Staging
environment: staging
secrets:
  NPM_TOKEN: ${{ secrets.NPM_TOKEN_STAGING }}
```

### 2. Secret Rotation Schedule

```markdown
| Secret | Rotation Frequency | Next Rotation |
|--------|-------------------|---------------|
| NPM_TOKEN | 90 days | 2025-02-10 |
```

### 3. Access Audit

```bash
# Check who has access to secrets
gh api repos/ruvnet/agentic-flow/collaborators

# Check recent workflow runs
gh run list --limit 10

# Check NPM audit log
npm audit
```

### 4. Emergency Revocation

If token is compromised:
```bash
# 1. Immediately revoke on NPM
# Visit: https://www.npmjs.com/settings/tokens

# 2. Delete GitHub secret
gh secret delete NPM_TOKEN

# 3. Generate new token
# Follow setup steps above

# 4. Audit recent publishes
npm view agentic-jujutsu versions
```

## Setup Checklist

- [ ] NPM account with publish permissions
- [ ] Generate NPM automation token
- [ ] Add NPM_TOKEN to GitHub secrets
- [ ] Verify GITHUB_TOKEN permissions
- [ ] Test build workflow
- [ ] Test publish workflow (dry run)
- [ ] Document token rotation date
- [ ] Set calendar reminder for rotation

## Support

If you encounter issues:
1. Check workflow logs: `gh run view [run-id]`
2. Verify secret configuration: `gh secret list`
3. Test locally with token: `NPM_TOKEN=... npm publish --dry-run`
4. Review NPM audit logs: https://www.npmjs.com/settings/audit-log

## References

- [NPM Token Documentation](https://docs.npmjs.com/creating-and-viewing-access-tokens)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub Token Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication)
